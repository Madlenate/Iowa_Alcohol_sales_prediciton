"""
Iowa Liquor Sales -- "Iowa's Liquor Epidemic" interactive Streamlit app.

Run with:
    streamlit run app.py

Ports the analysis from Project_1_analysis.ipynb into a dashboard, organized
the way the notebook tells the story: overview of the data -> explore the
raw trends -> PCA (why we cut features down / multicollinearity) ->
classification (what's associated with a sales dip) -> forecast (WIP).

Uses query_db.q(), so it reads from iowa_liquor.duckdb if present, or falls
back to the parquet files (for the `sales` table only -- the smaller
auxiliary tables like unemployment/homelessness/county_population/
alcohol_policy currently only exist in the local .duckdb file, so most of
this app needs that file present to run).
"""

import base64
import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from query_db import q

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Iowa's Liquor Epidemic", page_icon="\U0001F943", layout="wide")

PALETTE = {"blue": "#4C72B0", "red": "#C44E52", "green": "#55A868", "grey": "#808080"}


# ---------------------------------------------------------------------------
# Chrome: header banner + sidebar nav styling
# ---------------------------------------------------------------------------

@st.cache_data
def _flag_data_uri():
    svg_path = PROJECT_DIR / "assets" / "iowa_flag.svg"
    if not svg_path.exists():
        return None
    b64 = base64.b64encode(svg_path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def render_chrome():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 8px 12px;
            border-radius: 8px;
            transition: background-color 0.15s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background-color: rgba(191, 10, 48, 0.18);
            cursor: pointer;
        }
        .iowa-hero {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 14px 22px;
            margin-bottom: 18px;
            border-radius: 10px;
            background: linear-gradient(90deg, #002868 0%, #16305e 60%, #BF0A30 130%);
        }
        .iowa-hero img { height: 64px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.4); }
        .iowa-hero .kicker {
            color: #d8dfef; font-size: 0.8rem; letter-spacing: 0.25em;
            font-weight: 600; margin: 0;
        }
        .iowa-hero h1 { color: #ffffff; margin: 2px 0 0 0; font-size: 2rem; line-height: 1.15; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    flag_uri = _flag_data_uri()
    img_tag = f'<img src="{flag_uri}" alt="Flag of Iowa">' if flag_uri else ""
    st.markdown(
        f"""
        <div class="iowa-hero">
            {img_tag}
            <div>
                <p class="kicker">IOWA</p>
                <h1>Iowa's Liquor Epidemic</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading (cached so this only re-runs when the underlying data or
# query changes, not on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_data
def load_overview_meta():
    row_count = q("SELECT COUNT(*) AS n FROM sales").iloc[0]["n"]
    years = q("SELECT MIN(sale_year) AS mn, MAX(sale_year) AS mx, COUNT(DISTINCT county_name) AS counties FROM sales").iloc[0]
    sample = q("SELECT * FROM sales LIMIT 8")
    return {
        "rows": int(row_count),
        "year_min": int(years["mn"]),
        "year_max": int(years["mx"]),
        "counties": int(years["counties"]),
        "sample": sample,
    }


@st.cache_data
def load_statewide():
    """Statewide monthly sales + economic/demographic context. Mirrors notebook cell 37."""
    main_df = q("""
        SELECT
            sale_year,
            EXTRACT(month FROM ordered_on) AS month,
            SUM(sales_dollars) AS total_dollars,
            SUM(sales_bottles) AS total_bottles,
            COUNT(DISTINCT store_no) AS n_stores,
            COUNT(DISTINCT vendor_number) AS n_vendors
        FROM sales
        WHERE county_name IS NOT NULL AND sale_year < 2026
        GROUP BY 1, 2
    """)

    unemployment_df = q("""
        SELECT obs_year AS sale_year, obs_month AS month, unemployment_rate
        FROM unemployment
    """)

    homeless_df = q("SELECT * FROM homelessness")

    population_df = q("""
        SELECT pop_year AS sale_year, SUM(population) AS population
        FROM county_population
        GROUP BY 1
    """)

    df = main_df.merge(unemployment_df, on=["sale_year", "month"])
    df = df.merge(
        homeless_df[["pit_year", "overall_homeless", "overall_chronically_homeless", "unsheltered_homeless"]]
        .rename(columns={"pit_year": "sale_year"}),
        on="sale_year", how="left",
    )
    df = df.merge(population_df, on="sale_year", how="left")
    df = df.sort_values(["sale_year", "month"]).reset_index(drop=True)

    df["avg_price_per_bottle"] = df["total_dollars"] / df["total_bottles"]
    df["stores_per_10k_capita"] = df["n_stores"] / df["population"] * 10_000
    df["time_index"] = (df["sale_year"] - df["sale_year"].min()) * 12 + df["month"]
    df["log_dollars"] = np.log1p(df["total_dollars"])
    return df


@st.cache_data
def load_vendor_year(year):
    return q(f"""
        SELECT vendor_name,
            SUM(sales_bottles) AS bottles,
            SUM(sales_dollars) AS dollars
        FROM sales
        WHERE sale_year = {int(year)}
        GROUP BY vendor_name
    """)


@st.cache_data
def load_county_level(year):
    df = q(f"""
        SELECT county_name,
            SUM(sales_dollars) AS total_dollars,
            SUM(sales_liters) AS total_liters,
            COUNT(DISTINCT store_no) AS n_stores
        FROM sales
        WHERE county_name IS NOT NULL AND county_name != 'EL PASO' AND sale_year = {int(year)}
        GROUP BY county_name
    """)
    df["price_per_liter"] = df["total_dollars"] / df["total_liters"]
    return df


@st.cache_data
def load_pca_features():
    """County-month feature table used for PCA. Mirrors notebook cells 18-20."""
    cat_df = q("""
        SELECT
            county_name, sale_year, EXTRACT(month FROM ordered_on) AS month,
            category_name, SUM(sales_dollars) AS cat_dollars
        FROM sales
        WHERE county_name IS NOT NULL AND sale_year < 2026
        GROUP BY 1, 2, 3, 4
    """)
    main_df = q("""
        SELECT
            county_name, sale_year, EXTRACT(month FROM ordered_on) AS month,
            SUM(sales_dollars) AS total_dollars,
            SUM(sales_bottles) AS total_bottles,
            COUNT(DISTINCT store_no) AS n_stores,
            COUNT(DISTINCT vendor_number) AS n_vendors,
            AVG(pack) AS avg_pack,
            AVG(bottle_volume_ml) AS avg_bottle_volume_ml
        FROM sales
        WHERE county_name IS NOT NULL AND sale_year < 2026
        GROUP BY 1, 2, 3
    """)
    unemployment_df = q("SELECT obs_year AS sale_year, obs_month AS month, unemployment_rate FROM unemployment")
    homeless_df = q("SELECT * FROM homelessness")
    population_df = q("SELECT county_name, pop_year AS sale_year, population FROM county_population")

    grp_total = cat_df.groupby(["county_name", "sale_year", "month"])["cat_dollars"].transform("sum")
    cat_df["share"] = cat_df["cat_dollars"] / grp_total
    cat_df["p_logp"] = -cat_df["share"] * np.log(cat_df["share"].where(cat_df["share"] > 0))

    entropy = (
        cat_df.groupby(["county_name", "sale_year", "month"])["p_logp"]
        .sum().reset_index(name="category_entropy")
    )
    n_categories = (
        cat_df.groupby(["county_name", "sale_year", "month"])["category_name"]
        .nunique().reset_index(name="n_categories")
    )

    df = main_df.merge(entropy, on=["county_name", "sale_year", "month"])
    df = df.merge(n_categories, on=["county_name", "sale_year", "month"])
    df = df.merge(unemployment_df, on=["sale_year", "month"])
    df = df.merge(
        homeless_df[["pit_year", "overall_homeless", "overall_chronically_homeless", "unsheltered_homeless"]]
        .rename(columns={"pit_year": "sale_year"}),
        on="sale_year", how="left",
    )
    df = df.merge(population_df, on=["county_name", "sale_year"], how="left")
    df["avg_price_per_bottle"] = df["total_dollars"] / df["total_bottles"]
    df["stores_per_10k_capita"] = df["n_stores"] / df["population"] * 10_000
    return df


FEATURE_COLS = [
    "n_stores", "n_vendors", "n_categories",
    "avg_pack", "avg_bottle_volume_ml", "avg_price_per_bottle",
    "category_entropy", "unemployment_rate",
    "overall_homeless", "overall_chronically_homeless", "unsheltered_homeless", "stores_per_10k_capita",
]


@st.cache_data
def run_pca():
    df = load_pca_features()
    X = df[FEATURE_COLS].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pcs = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=FEATURE_COLS,
        columns=[f"PC{i + 1}" for i in range(len(FEATURE_COLS))],
    )
    corr = X.corr()
    return {
        "evr": pca.explained_variance_ratio_,
        "cum": np.cumsum(pca.explained_variance_ratio_),
        "loadings": loadings,
        "corr": corr,
        "pcs": pcs,
    }


@st.cache_data
def load_classification():
    """Baseline residual -> dip label -> logistic regression. Mirrors the notebook's
    classification cells (the ones right after the ABV-band policy merge)."""
    df = load_statewide().copy()
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)

    # Step 1: baseline model (n_stores + price + month) -> residual, used as an
    # "embedding" that strips out the seasonal/structural part of sales so what's
    # left (the residual) reflects the unusual, month-to-month swings.
    baseline_cols = ["n_stores", "avg_price_per_bottle"]
    baseline_data = pd.concat([df[baseline_cols + ["log_dollars"]], month_dummies], axis=1).dropna()
    X_base = baseline_data[baseline_cols + list(month_dummies.columns)]
    y_base = baseline_data["log_dollars"]
    baseline_model = LinearRegression().fit(X_base, y_base)
    df.loc[baseline_data.index, "residual"] = y_base - baseline_model.predict(X_base)

    # Step 2: rule-based "dip" label -- a SUSTAINED downturn, not a one-off bad
    # month. A centered 3-month rolling average residual below -0.5 SD filters
    # out single-month noise.
    resid_std = df["residual"].std()
    df["rolling_resid"] = df["residual"].rolling(3, center=True).mean()
    df["dip"] = (df["rolling_resid"] < -0.5 * resid_std).astype("Int64")

    # Step 3: bring in the 2023 retail-licensing policy flag and fit the classifier.
    policy_df = q("SELECT policy_year AS sale_year, post_2019_spirits_deregulation, post_2023_spirits_expansion FROM alcohol_policy")
    df = df.merge(policy_df, on="sale_year", how="left")

    clf_cols = ["n_stores", "n_vendors", "avg_price_per_bottle", "unemployment_rate",
                "overall_homeless", "unsheltered_homeless", "post_2023_spirits_expansion"]
    clf_data = pd.concat([df[clf_cols + ["dip"]], month_dummies], axis=1).dropna()
    X_clf = clf_data[clf_cols + list(month_dummies.columns)]
    y_clf = clf_data["dip"].astype(int)

    log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    log_reg.fit(X_clf, y_clf)
    y_pred = log_reg.predict(X_clf)
    report = classification_report(y_clf, y_pred, target_names=["normal", "dip"], output_dict=True)
    cm = confusion_matrix(y_clf, y_pred)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(log_reg, X_clf, y_clf, cv=skf, scoring="f1")

    coefs = pd.Series(
        log_reg.named_steps["logisticregression"].coef_[0][: len(clf_cols)], index=clf_cols
    ).sort_values(key=abs, ascending=False)

    return {
        "df": df,
        "report": report,
        "confusion_matrix": cm,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "coefs": coefs,
        "n_dip_months": int(df["dip"].sum()),
        "n_labeled_months": int(df["dip"].notna().sum()),
    }


# ---------------------------------------------------------------------------
# Forecast helpers (kept for when the modeling approach below gets revisited --
# see the "Forecast" page). The lag-12 index bug from the previous version is
# fixed here: it read `idx_last - 11 + h`, which pulled a value 11 months back
# instead of 12, shifting every seasonal anchor by a month. Fixed to `idx_last
# - 12 + h`, verified against the notebook's cell-49 forecast numbers.
# ---------------------------------------------------------------------------

@st.cache_data
def state_level_diagnostics():
    """CV R^2 and naive-baseline R^2 for the state-level Lasso model, for the metrics
    summary table. Not cached on build_forecast_model itself since that takes a
    DataFrame argument Streamlit can't hash reliably -- computed fresh here instead."""
    statewide = load_statewide()
    df, model, forecast_cols, fc_data, X_forecast = build_forecast_model(statewide)
    y_forecast = fc_data["log_dollars_smooth_next"]
    tscv = TimeSeriesSplit(n_splits=5)
    model_scores, naive_scores = [], []
    for train_idx, test_idx in tscv.split(X_forecast):
        m = make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=10000))
        m.fit(X_forecast.iloc[train_idx], y_forecast.iloc[train_idx])
        pred = m.predict(X_forecast.iloc[test_idx])
        model_scores.append(r2_score(y_forecast.iloc[test_idx], pred))
        naive_scores.append(r2_score(y_forecast.iloc[test_idx], fc_data["log_dollars_smooth_lag12"].iloc[test_idx]))
    return {"r2": np.mean(model_scores), "r2_std": np.std(model_scores), "naive_r2": np.mean(naive_scores)}


def build_forecast_model(df):
    df = df.copy()
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)

    baseline_cols = ["n_stores", "avg_price_per_bottle"]
    baseline_data = pd.concat([df[baseline_cols], month_dummies], axis=1)
    baseline_model = LinearRegression().fit(baseline_data, df["log_dollars"])
    df["residual"] = df["log_dollars"] - baseline_model.predict(baseline_data)

    df["residual_lag1"] = df["residual"].shift(1)
    df["n_stores_lag1"] = df["n_stores"].shift(1)
    df["price_lag1"] = df["avg_price_per_bottle"].shift(1)

    df["log_dollars_smooth"] = df["log_dollars"].rolling(3).mean()
    df["log_dollars_smooth_next"] = df["log_dollars_smooth"].shift(-1)
    df["log_dollars_smooth_lag12"] = df["log_dollars_smooth"].shift(11)

    forecast_cols = ["log_dollars_smooth_lag12", "residual_lag1", "n_stores_lag1", "price_lag1"]
    fc_data = df[forecast_cols + ["log_dollars_smooth_next"]].dropna()
    X_forecast = fc_data[forecast_cols]
    y_forecast = fc_data["log_dollars_smooth_next"]

    model = make_pipeline(StandardScaler(), Lasso(alpha=0.01))
    model.fit(X_forecast, y_forecast)

    return df, model, forecast_cols, fc_data, X_forecast


def multi_month_forecast(df, model, forecast_cols, horizons=3):
    """Recursive forecast. residual/n_stores/price are carried forward at their last
    known level past h=1 (not separately forecasted); only the lag-12 seasonal anchor
    uses real historical data at every horizon."""
    idx_last = df.index[-1]
    last_month = int(df.loc[idx_last, "month"])
    rows = []
    for h in range(1, horizons + 1):
        target_month = 1 if (last_month + h - 1) % 12 == 0 else (last_month + h - 1) % 12 + 1
        target_year = int(df.loc[idx_last, "sale_year"]) + (last_month + h - 1) // 12
        target_date = pd.Timestamp(year=target_year, month=target_month, day=1)

        step_features = {
            "log_dollars_smooth_lag12": df["log_dollars_smooth"].iloc[idx_last - 12 + h],
            "residual_lag1": df["residual"].iloc[idx_last],
            "n_stores_lag1": df["n_stores"].iloc[idx_last],
            "price_lag1": df["avg_price_per_bottle"].iloc[idx_last],
        }
        step_X = pd.DataFrame([step_features])[forecast_cols]
        step_dollars = np.expm1(model.predict(step_X)[0])
        rows.append((target_date, step_dollars))
    return pd.DataFrame(rows, columns=["date", "forecast_dollars"])


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def page_overview():
    st.header("Overview")

    st.subheader("Our goal")
    st.markdown(
        "1. **Find associative features** behind why liquor sales in Iowa dip or grow "
        "month to month -- store counts, pricing, unemployment, homelessness, policy changes.\n"
        "2. **Use that insight to look ahead** -- take what's associated with past ups and "
        "downs and use it to get a sense of what sales might look like going forward."
    )

    st.subheader("Why this matters")
    st.markdown(
        "Finding associative drivers isn't just an academic exercise -- the intent is to give a "
        "public health agency, a community organization, or state/local government a starting list "
        "of levers worth investigating if the goal is reducing alcohol sales and the harm that comes "
        "with heavy drinking. If store density, pricing, or a specific policy change are genuinely "
        "associated with higher sales, those are the kinds of things a regulator or advocacy group "
        "could act on -- capping license density, adjusting pricing policy, or targeting outreach in "
        "counties where the data shows the strongest associations. This dashboard doesn't make that "
        "case on its own -- see **Scope & limitations** below, and **What Drives a Sales Dip** for "
        "why these are associations, not proof -- but it's meant to be a starting point for that "
        "kind of investigation, not just a sales dashboard."
    )

    st.subheader("Scope & limitations")
    st.markdown(
        "Worth being upfront about what this data actually is, since it directly limits what "
        "\"forecasting alcohol sales\" can mean here. This is **wholesale purchase data** -- "
        "records of what a *retail store* bought from its distributor (`store_no`, "
        "`vendor_number`, `pack`, `invoice_id`) -- not point-of-sale data of what a *customer* "
        "bought at the register. There's no customer identifier, no receipt-level basket, no time "
        "of day, no promotion flag -- nothing that captures an actual consumer purchase decision.\n\n"
        "That means the signal available here is almost entirely **supply-side and structural**: "
        "how many stores are open, what things cost, how sales move with the calendar. It is not "
        "**consumer-behavior signal** -- why any particular person bought a bottle that month. A "
        "store's wholesale restocking can also lag or lead actual retail demand -- a spike here "
        "might reflect stores stocking up ahead of a price or policy change rather than a genuine "
        "surge in drinking, and there's no way to tell those apart with what's in this table.\n\n"
        "Practically, this shapes every model in this dashboard: aggregated to the statewide level, "
        "structural signal (store counts, pricing, seasonality) is strong enough to explain real "
        "variance in monthly sales. Pushed down to an individual county's month-to-month movement, "
        "that structural signal runs out -- testing that gap directly is exactly what the "
        "county-level detail further down this section, and the Forecast section, walk through."
    )

    meta = load_overview_meta()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sales transactions", f"{meta['rows']:,}")
    col2.metric("Years covered", f"{meta['year_min']}-{meta['year_max']}")
    col3.metric("Counties", meta["counties"])
    col4.metric("On-disk size", "~544 MB")

    st.subheader("Where the data comes from")
    st.markdown(
        """
| Dataset | What it is | Source |
|---|---|---|
| Liquor sales | Every wholesale liquor purchase transaction by Iowa retailers, 2012-2026 | Iowa's open data portal (Iowa Alcoholic Beverages Division), data.iowa.gov |
| Homelessness | Annual statewide Point-in-Time homelessness counts for Iowa | HUD Exchange HDX -- https://www.hudexchange.info/programs/hdx/ |
| Unemployment | National monthly unemployment rate | FRED (Federal Reserve Bank of St. Louis) -- https://fred.stlouisfed.org/ |
| County population | Annual county-level population estimates | Iowa public population records |
| Alcohol policy | Iowa's retail/wholesale alcohol licensing rules by year (incl. the 2023 licensing overhaul) | Iowa public policy records |
| County health | County-level health outcome measures | CDC PLACES |
        """
    )

    st.subheader("How it's stored and queried")
    st.markdown(
        "The raw CSVs (~7 GB) get loaded into **DuckDB** once (`Data_Base_creation.py`), "
        "then exported to **Parquet**, partitioned by year (`parquet/by_year/sale_year=YYYY/`). "
        "Every query in this app goes through `query_db.q()`, which opens the local "
        "`iowa_liquor.duckdb` file if it's present, or otherwise runs directly against the "
        "Parquet files -- DuckDB reads Parquet natively, so no server or import step is needed "
        "to explore ~36 million rows interactively."
    )

    st.subheader("A look at the raw data")
    st.dataframe(meta["sample"], use_container_width=True)


def page_investigate():
    st.header("Investigate & Understand the Data")
    st.markdown(
        "This section explores the raw data before any modeling happens: sales trends by vendor "
        "and month, how sales compare against unemployment, homelessness, and population over time, "
        "and a closer look at what changes when you break the same numbers down to the county level "
        "instead of one statewide total."
    )

    statewide = load_statewide()
    years = sorted(statewide["sale_year"].unique())

    st.subheader("Vendors and monthly sales, by year")
    year = st.selectbox("Year", years, index=len(years) - 1)

    monthly = statewide[statewide["sale_year"] == year].sort_values("month")
    monthly_millions = monthly["total_dollars"] / 1_000_000
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar([calendar.month_abbr[m] for m in monthly["month"]], monthly_millions, color=PALETTE["blue"])
    ax.bar_label(ax.containers[0], fmt="${:,.1f}M", padding=3, fontsize=9)
    ax.set_ylim(0, monthly_millions.max() * 1.15)
    ax.set_ylabel("sales ($ millions)")
    ax.set_title(f"Iowa liquor sales by month -- {year}")
    st.pyplot(fig)

    vendors = load_vendor_year(year)
    top_bottles = vendors.nlargest(10, "bottles").set_index("vendor_name")["bottles"].div(1_000_000).sort_values()
    top_dollars = vendors.nlargest(10, "dollars").set_index("vendor_name")["dollars"].div(1_000_000).sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.barh(top_bottles.index, top_bottles.values, color=PALETTE["blue"])
    ax1.set_title("Top vendors by bottles sold")
    ax1.set_xlabel("bottles (millions)")
    ax2.barh(top_dollars.index, top_dollars.values, color=PALETTE["red"])
    ax2.set_title("Top vendors by revenue")
    ax2.set_xlabel("sales ($ millions)")
    fig.suptitle(f"Iowa liquor vendors -- {year}")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Statewide context, 2012-2025")
    plot_df = statewide.sort_values(["sale_year", "month"]).copy()
    plot_df["date"] = pd.to_datetime(dict(year=plot_df["sale_year"], month=plot_df["month"], day=1))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(plot_df["date"], plot_df["unemployment_rate"], color=PALETTE["blue"])
        ax.set_title("U.S. national unemployment rate")
        ax.set_ylabel("rate (%)")
        st.pyplot(fig)
    with col2:
        homeless_yearly = plot_df.drop_duplicates("sale_year")[["sale_year", "overall_homeless"]].dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(homeless_yearly["sale_year"], homeless_yearly["overall_homeless"], color=PALETTE["red"], marker="o")
        ax.set_title("Iowa overall homelessness (HUD PIT count)")
        ax.set_ylabel("people counted")
        st.pyplot(fig)

    population_df = q("SELECT pop_year AS sale_year, SUM(population) AS population FROM county_population GROUP BY 1").sort_values("sale_year")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(population_df["sale_year"], population_df["population"] / 1_000_000, color=PALETTE["green"], marker="o")
    ax.set_title("Iowa population (counties present in our sales data)")
    ax.set_ylabel("population (millions)")
    st.pyplot(fig)

    st.subheader("Everything indexed together")
    st.caption("Each series set to 100 at its first month, so very different scales (dollars, headcounts, a rate) become comparable growth lines.")
    indexed_monthly = plot_df.set_index("date")[
        ["total_dollars", "n_stores", "n_vendors", "avg_price_per_bottle", "unemployment_rate", "overall_homeless", "stores_per_10k_capita"]
    ]
    indexed_monthly = indexed_monthly / indexed_monthly.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in indexed_monthly.columns:
        ax.plot(indexed_monthly.index, indexed_monthly[col], label=col)
    ax.axhline(100, color="grey", lw=0.5)
    ax.set_ylabel(f"index (100 = {plot_df['date'].iloc[0]:%b %Y})")
    ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig)

    st.subheader("A closer look: county-level detail")

    county_view = st.radio(
        "View", ["Sales ($)", "Price per liter ($)"], horizontal=True, label_visibility="collapsed"
    )

    county_year = load_county_level(year)

    if county_view == "Sales ($)":
        st.markdown(
            "Zooming into individual counties for the year selected above shows how concentrated Iowa's "
            "liquor sales really are -- a handful of populous counties account for a large share of the "
            "statewide total, tracking store count (and population) far more than anything else."
        )

        top_counties = county_year.nlargest(15, "total_dollars").sort_values("total_dollars")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        sales_millions = top_counties["total_dollars"] / 1_000_000
        ax1.barh(top_counties["county_name"], sales_millions, color=PALETTE["blue"])
        ax1.set_title(f"Top 15 counties by sales -- {year}")
        ax1.set_xlabel("sales ($ millions)")
        ax2.barh(top_counties["county_name"], top_counties["n_stores"], color=PALETTE["red"])
        ax2.set_title(f"Store count, same counties -- {year}")
        ax2.set_xlabel("unique stores")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Could county-level detail improve the forecast?")
        st.markdown(
            "One idea we tested directly: instead of forecasting from a single statewide number "
            "(168 monthly rows, 2012-2025), train on a county-month panel instead -- roughly 15,000 "
            "rows, since each of Iowa's ~99 counties gets its own row every month. More rows should mean "
            "more material for a model to learn from.\n\n"
            "**It didn't hold up.** The headline cross-validated R² for the county-panel model looked "
            "great at first glance (0.987) -- but a baseline that isn't even a model (just guessing each "
            "county's value from the same month one year earlier, no fitting at all) already scored "
            "0.970. Almost all of that 0.987 was really just \"Polk County is always huge and a rural "
            "county is always tiny,\" which any predictor nails instantly once county sizes span two "
            "orders of magnitude. Once we isolated the actual question -- can this predict whether a "
            "*given* county moves up or down relative to its own normal level -- the real (\"within-"
            "county\") R² came back **negative** (roughly -1.3), meaning it did worse than"
            "assuming each county keeps on just staying small and having small sales.\n\n"
        )

    else:  # Price per liter ($)
        top_counties = county_year.nlargest(15, "total_dollars").sort_values("total_dollars")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        ax1.barh(top_counties["county_name"], top_counties["price_per_liter"], color=PALETTE["green"])
        ax1.set_title(f"Price per liter, same counties -- {year}")
        ax1.set_xlabel("price per liter ($)")
        ax2.barh(top_counties["county_name"], top_counties["n_stores"], color=PALETTE["red"])
        ax2.set_title(f"Store count, same counties -- {year}")
        ax2.set_xlabel("unique stores")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Why look at price per liter?")
        st.markdown(
            "`total_dollars` scales almost entirely with county size -- Polk moves ~800x the dollars "
            "of the smallest county, simply because it has ~800x the population and stores. That size "
            "gap dominates any model trained on raw totals, drowning out whatever genuine month-to-"
            "month signal exists underneath it.\n\n"
            "`price_per_liter` (`total_dollars / total_liters`) is different: it's a *rate*, not a "
            "*total*, so it doesn't automatically scale with county size the way raw dollars do -- a "
            "bottle of vodka costs roughly the same in Polk as it does in a small rural county. Our "
            "methodology going into this: if we can normalize away the county-size effect at the "
            "source, by predicting a rate instead of a total, a county-panel model might get a fairer "
            "shot at learning genuine temporal patterns instead of just re-deriving each county's size "
            "for free.\n\n"
            "We tested this directly, the same way we tested the raw-dollar version -- results (what "
            "held up and what didn't) are written up in the **Forecast** section rather than here, "
            "since it's a forecasting question and that's where it belongs."
        )


def page_pca():
    st.header("PCA & Multicollinearity")
    st.markdown(
        "Before modeling anything, we ran PCA on the county-month feature table to see "
        "how much of the variance across our candidate features could be explained by a "
        "handful of components -- and, just as importantly, whether the features were "
        "actually independent signals or just restating each other."
    )

    result = run_pca()
    evr, cum, loadings, corr = result["evr"], result["cum"], result["loadings"], result["corr"]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(corr.shape[1])); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(corr.shape[0])); ax.set_yticklabels(corr.index, fontsize=7)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="correlation")
        ax.set_title("Feature correlation matrix")
        fig.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(loadings.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(loadings.shape[1])); ax.set_xticklabels(loadings.columns, fontsize=7)
        ax.set_yticks(range(loadings.shape[0])); ax.set_yticklabels(loadings.index, fontsize=7)
        for i in range(loadings.shape[0]):
            for j in range(loadings.shape[1]):
                ax.text(j, i, f"{loadings.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="loading")
        ax.set_title("PCA component loadings")
        fig.tight_layout()
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(1, len(evr) + 1), evr, color=PALETTE["blue"], label="individual")
    ax.plot(range(1, len(evr) + 1), cum, color=PALETTE["red"], marker="o", label="cumulative")
    ax.set_xlabel("principal component"); ax.set_ylabel("explained variance ratio")
    ax.set_title("Scree plot"); ax.set_ylim(0, 1.05); ax.legend()
    st.pyplot(fig)

    # EDIT ME: this is the explanation to expand/rewrite -- multicollinearity findings,
    # which features turned out to be redundant, and what that meant for feature
    # selection going into the classification model below.
    pca_explanation = """
Even though **PC1** captured the largest share of explained variance, it also surfaced a real
problem with the county-month feature set: several of these features move together closely
enough that they're not contributing independent information. That's **multicollinearity** --
when input features are highly correlated with each other, a model can't cleanly attribute
effect to one feature over another, and the coefficients it produces become unstable and hard
to trust. Looking at the correlation matrix above, `n_stores` and `stores_per_10k_capita`, along
with a few of the entropy/category-count features, load onto the same components -- they're
largely restating store density in different units rather than adding new signal. Rather than
feed every feature into a downstream model, we used this to prune down to the features that
carry distinct signal, which is what feeds the classification step below.
    """
    st.markdown(pca_explanation)


def page_classification():
    st.header("What Drives a Sales Dip")
    st.markdown(
        "This section asks a narrower question than \"what drives sales\": **what's "
        "associated with a sustained *downturn* in sales?** Important distinction up front -- "
        "everything below is **associative, not causal**. A logistic regression coefficient "
        "tells us a feature moves together with dip months in our historical data; it doesn't "
        "prove that feature *causes* the dip. Treat these as leads worth investigating, not "
        "conclusions."
    )

    st.subheader("Step 1 -- residual embedding")
    st.markdown(
        "We first fit a plain linear regression of `log(sales)` on `n_stores`, "
        "`avg_price_per_bottle`, and month-of-year dummies. That model captures the "
        "*structural* part of sales -- how many stores are open, what things cost, and "
        "normal seasonality. What's left over (**actual minus predicted**, the *residual*) is "
        "the part of each month's sales that structure alone doesn't explain -- an embedding "
        "of the unusual months, good or bad."
    )

    st.subheader("Step 2 -- labeling a \"dip\"")
    st.markdown(
        "A single bad month is noise. We flag a **dip** only when the centered 3-month "
        "rolling average of that residual drops more than 0.5 standard deviations below zero "
        "-- a sustained downturn, not a one-off."
    )

    result = load_classification()
    st.caption(f"{result['n_dip_months']} months flagged as a dip out of {result['n_labeled_months']} labeled.")

    fig, ax = plt.subplots(figsize=(12, 4))
    df = result["df"].sort_values(["sale_year", "month"]).copy()
    df["date"] = pd.to_datetime(dict(year=df["sale_year"], month=df["month"], day=1))
    ax.plot(df["date"], df["residual"], color=PALETTE["grey"], alpha=0.5, label="monthly residual")
    ax.plot(df["date"], df["rolling_resid"], color=PALETTE["blue"], label="3-month rolling residual")
    dips = df[df["dip"] == 1]
    ax.scatter(dips["date"], dips["rolling_resid"], color=PALETTE["red"], zorder=5, label="flagged dip")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Sales residual over time, with flagged dip months")
    st.pyplot(fig)

    st.subheader("Step 3 -- what's associated with a dip")
    st.markdown(
        "We trained a logistic regression to predict the dip label from `n_stores`, "
        "`n_vendors`, `avg_price_per_bottle`, `unemployment_rate`, `overall_homeless`, "
        "`unsheltered_homeless`, and a flag for **post-2023 spirits licensing expansion**, "
        "plus month dummies (class-balanced, since dips are the minority class)."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        coefs = result["coefs"]
        colors = [PALETTE["red"] if v < 0 else PALETTE["blue"] for v in coefs.values]
        ax.barh(coefs.index[::-1], coefs.values[::-1], color=colors[::-1])
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title("Standardized logistic regression coefficients")
        ax.set_xlabel("effect on log-odds of a dip")
        st.pyplot(fig)
    with col2:
        st.metric("Cross-validated F1 (dip class)", f"{result['cv_f1_mean']:.2f}", f"± {result['cv_f1_std']:.2f}")
        dip_f1 = result["report"]["dip"]["f1-score"]
        st.metric("In-sample F1 (dip class)", f"{dip_f1:.2f}")

    st.subheader("Reading the associations")
    st.markdown(
        """
A few examples of how to read these coefficients as *associative* stories, not proven causes.
Remember the sign here is about the **dip label**, not the sales level -- a dip is a sustained
downturn in the *residual* left over after a baseline model already accounts for store count,
price, and seasonality. So a feature can be positively associated with dips while the underlying
sales level is still growing overall; the two aren't the same question.

- **The 2023 licensing overhaul.** Iowa's 2023 retail alcohol licensing overhaul streamlined
  multiple overlapping permits into a single Class E Retail Alcohol License, making it easier
  for local businesses to operate and is widely credited with helping drive higher overall state
  liquor sales. In this model, `post_2023_spirits_expansion` actually comes back with a
  **positive** coefficient -- associated with *higher* odds of a dip month, not lower. That's not
  necessarily a contradiction: a policy shock that expands access can still coincide with more
  residual volatility while the market adjusts (new licensees ramping up unevenly, pricing
  shifting), even as the overall sales level rises. It's a reminder that "grew the top line" and
  "grew more volatile" can both be true at once.
- **Homelessness.** `overall_homeless` is also positively associated with dip odds here. One
  reading ties back to the mental-health/substance-use angle -- economic distress could show up
  as pulled-back discretionary spending rather than more drinking. But homelessness has also
  trended over the same years as the rest of the dataset, so this could just as easily be picking
  up a shared time trend rather than a direct behavioral link. Notably, `unsheltered_homeless`
  points the *other* direction (negative) -- the two homelessness measures don't even agree with
  each other, which is exactly the kind of thing that should make you skeptical of a tidy causal
  story here.
- **Price and store count.** `avg_price_per_bottle` and `n_stores` are both positively associated
  with dip odds too -- worth investigating further rather than reading at face value, since both
  also trend upward over the dataset's timespan.

None of this tells us direction of causality -- for that we'd need a designed study, not
historical correlations, and several of these signs look like they could be confounded with the
overall time trend rather than a direct mechanism. What it does give us is a short list of
variables worth investigating further, and it's the same residual-embedding idea from Step 1 that
we intended to carry into the forecasting step next.
        """
    )


@st.cache_data
def load_county_panel():
    panel = q("""
        SELECT county_name, sale_year, EXTRACT(month FROM ordered_on) AS month,
            SUM(sales_dollars) AS total_dollars,
            SUM(sales_bottles) AS total_bottles,
            SUM(sales_liters) AS total_liters,
            COUNT(DISTINCT store_no) AS n_stores
        FROM sales
        WHERE county_name IS NOT NULL AND county_name != 'EL PASO' AND sale_year < 2026
        GROUP BY 1, 2, 3
    """)
    panel["avg_price_per_bottle"] = panel["total_dollars"] / panel["total_bottles"]
    panel["price_per_liter"] = panel["total_dollars"] / panel["total_liters"]
    panel["log_dollars"] = np.log1p(panel["total_dollars"])
    return panel.sort_values(["county_name", "sale_year", "month"]).reset_index(drop=True)


def _cv_diagnostics(X_all, y_all, fc_panel, naive_col):
    """Pooled CV R^2, a trivial same-county-last-year naive baseline, and within-county
    (demeaned) R^2 -- the three numbers needed to tell real skill apart from the
    between-county size effect. See the Investigate page for the full explanation."""
    periods = fc_panel[["sale_year", "month"]].drop_duplicates().sort_values(["sale_year", "month"]).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=5)
    period_key = list(zip(fc_panel["sale_year"], fc_panel["month"]))
    model_scores, naive_scores, within_scores = [], [], []
    for train_p_idx, test_p_idx in tscv.split(periods):
        train_periods = set(map(tuple, periods.iloc[train_p_idx][["sale_year", "month"]].values))
        test_periods = set(map(tuple, periods.iloc[test_p_idx][["sale_year", "month"]].values))
        train_mask = np.array([k in train_periods for k in period_key])
        test_mask = np.array([k in test_periods for k in period_key])

        cv_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        cv_model.fit(X_all[train_mask], y_all[train_mask])
        pred = cv_model.predict(X_all[test_mask])
        model_scores.append(r2_score(y_all[test_mask], pred))
        naive_scores.append(r2_score(y_all[test_mask], fc_panel.loc[test_mask, naive_col]))

        county_means = y_all[train_mask].groupby(fc_panel.loc[train_mask, "county_name"]).mean()
        overall_mean = y_all[train_mask].mean()
        y_train_dm = y_all[train_mask] - fc_panel.loc[train_mask, "county_name"].map(county_means)
        y_test_dm = y_all[test_mask] - fc_panel.loc[test_mask, "county_name"].map(county_means).fillna(overall_mean)
        cv_model2 = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        cv_model2.fit(X_all[train_mask], y_train_dm)
        pred_dm = cv_model2.predict(X_all[test_mask])
        within_scores.append(r2_score(y_test_dm, pred_dm))

    return {
        "pooled_r2": np.mean(model_scores), "pooled_r2_std": np.std(model_scores),
        "naive_r2": np.mean(naive_scores),
        "within_r2": np.mean(within_scores), "within_r2_std": np.std(within_scores),
    }


@st.cache_data
def build_county_dollar_forecast():
    """County-month panel forecast of total_dollars, summed to a statewide total.
    Mirrors the notebook's county-panel forecasting experiment."""
    panel = load_county_panel().copy()
    month_dummies = pd.get_dummies(panel["month"], prefix="month", drop_first=True)
    baseline_cols = ["n_stores", "avg_price_per_bottle"]
    baseline_data = pd.concat([panel[baseline_cols + ["log_dollars"]], month_dummies], axis=1).dropna()
    X_base = baseline_data[baseline_cols + list(month_dummies.columns)]
    y_base = baseline_data["log_dollars"]
    baseline_model = LinearRegression().fit(X_base, y_base)
    panel.loc[baseline_data.index, "residual"] = y_base - baseline_model.predict(X_base)

    g = panel.groupby("county_name")
    panel["log_dollars_smooth"] = g["log_dollars"].transform(lambda s: s.rolling(3).mean())
    panel["log_dollars_smooth_next"] = g["log_dollars_smooth"].shift(-1)
    panel["log_dollars_smooth_lag12"] = g["log_dollars_smooth"].shift(11)
    panel["residual_lag1"] = g["residual"].shift(1)
    panel["n_stores_lag1"] = g["n_stores"].shift(1)
    panel["price_lag1"] = g["avg_price_per_bottle"].shift(1)

    forecast_cols = ["log_dollars_smooth_lag12", "residual_lag1", "n_stores_lag1", "price_lag1"]
    fc_panel = panel[["county_name", "sale_year", "month"] + forecast_cols + ["log_dollars_smooth_next"]].dropna()

    X_all = fc_panel[forecast_cols]
    y_all = fc_panel["log_dollars_smooth_next"]
    final_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    final_model.fit(X_all, y_all)

    lookup = panel.set_index(["county_name", "sale_year", "month"])
    last_rows = panel.sort_values(["county_name", "sale_year", "month"]).groupby("county_name").tail(1)

    horizons = 3
    rows = []
    for _, last in last_rows.iterrows():
        county = last["county_name"]
        last_year, last_month = int(last["sale_year"]), int(last["month"])
        for h in range(1, horizons + 1):
            target_month = 1 if (last_month + h - 1) % 12 == 0 else (last_month + h - 1) % 12 + 1
            target_year = last_year + (last_month + h - 1) // 12
            anchor_key = (county, target_year - 1, target_month)
            if anchor_key not in lookup.index or pd.isna(lookup.loc[anchor_key, "log_dollars_smooth"]):
                continue
            step_X = pd.DataFrame([{
                "log_dollars_smooth_lag12": lookup.loc[anchor_key, "log_dollars_smooth"],
                "residual_lag1": last["residual"],
                "n_stores_lag1": last["n_stores"],
                "price_lag1": last["avg_price_per_bottle"],
            }])[forecast_cols]
            pred_log = final_model.predict(step_X)[0]
            rows.append((county, target_year, target_month, h, np.expm1(pred_log)))

    county_forecasts = pd.DataFrame(rows, columns=["county_name", "sale_year", "month", "h", "forecast_dollars"])
    statewide_forecast = county_forecasts.groupby(["sale_year", "month"])["forecast_dollars"].sum().reset_index()
    statewide_forecast["date"] = pd.to_datetime(dict(year=statewide_forecast["sale_year"], month=statewide_forecast["month"], day=1))

    actual_statewide = panel.groupby(["sale_year", "month"])["total_dollars"].sum().reset_index()
    actual_statewide["date"] = pd.to_datetime(dict(year=actual_statewide["sale_year"], month=actual_statewide["month"], day=1))

    diagnostics = _cv_diagnostics(X_all, y_all, fc_panel, "log_dollars_smooth_lag12")
    return actual_statewide, statewide_forecast, diagnostics


@st.cache_data
def build_county_price_forecast():
    """County-month panel forecast of price_per_liter, averaged to a statewide figure
    (a rate, so averaged rather than summed -- unlike the dollar version above)."""
    panel = load_county_panel().copy()
    g = panel.groupby("county_name")
    panel["price_smooth"] = g["price_per_liter"].transform(lambda s: s.rolling(3).mean())
    panel["price_smooth_next"] = g["price_smooth"].shift(-1)
    panel["price_smooth_lag12"] = g["price_smooth"].shift(11)
    panel["price_lag1"] = g["price_per_liter"].shift(1)
    panel["n_stores_lag1"] = g["n_stores"].shift(1)

    forecast_cols = ["price_smooth_lag12", "price_lag1", "n_stores_lag1"]
    fc_panel = panel[["county_name", "sale_year", "month"] + forecast_cols + ["price_smooth_next"]].dropna()

    X_all = fc_panel[forecast_cols]
    y_all = fc_panel["price_smooth_next"]
    final_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    final_model.fit(X_all, y_all)

    lookup = panel.set_index(["county_name", "sale_year", "month"])
    last_rows = panel.sort_values(["county_name", "sale_year", "month"]).groupby("county_name").tail(1)

    horizons = 3
    rows = []
    for _, last in last_rows.iterrows():
        county = last["county_name"]
        last_year, last_month = int(last["sale_year"]), int(last["month"])
        for h in range(1, horizons + 1):
            target_month = 1 if (last_month + h - 1) % 12 == 0 else (last_month + h - 1) % 12 + 1
            target_year = last_year + (last_month + h - 1) // 12
            anchor_key = (county, target_year - 1, target_month)
            if anchor_key not in lookup.index or pd.isna(lookup.loc[anchor_key, "price_smooth"]):
                continue
            step_X = pd.DataFrame([{
                "price_smooth_lag12": lookup.loc[anchor_key, "price_smooth"],
                "price_lag1": last["price_per_liter"],
                "n_stores_lag1": last["n_stores"],
            }])[forecast_cols]
            pred_price = final_model.predict(step_X)[0]
            rows.append((county, target_year, target_month, h, pred_price))

    county_forecasts = pd.DataFrame(rows, columns=["county_name", "sale_year", "month", "h", "forecast_price_per_liter"])
    statewide_forecast = county_forecasts.groupby(["sale_year", "month"])["forecast_price_per_liter"].mean().reset_index()
    statewide_forecast["date"] = pd.to_datetime(dict(year=statewide_forecast["sale_year"], month=statewide_forecast["month"], day=1))

    actual_statewide = panel.groupby(["sale_year", "month"])["price_per_liter"].mean().reset_index()
    actual_statewide["date"] = pd.to_datetime(dict(year=actual_statewide["sale_year"], month=actual_statewide["month"], day=1))

    diagnostics = _cv_diagnostics(X_all, y_all, fc_panel, "price_smooth_lag12")
    return actual_statewide, statewide_forecast, diagnostics


def page_forecast():
    st.header("Forecast")
    st.info(
        "The original single-series state-level approach (Lasso on a 3-month-smoothed target) "
        "has a fixed lag-alignment bug but isn't wired up below -- what's shown instead is the "
        "county-panel experiment from the Investigate page, taken all the way through to an "
        "actual forward forecast, with the same honest diagnostics attached."
    )

    st.subheader("County-panel forecast: sales ($)")
    actual_d, forecast_d, diag_d = build_county_dollar_forecast()
    last_point_d = actual_d.iloc[[-1]][["date", "total_dollars"]].rename(columns={"total_dollars": "forecast_dollars"})
    connected_d = pd.concat([last_point_d, forecast_d[["date", "forecast_dollars"]]], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_d["date"].tail(36), actual_d["total_dollars"].tail(36) / 1_000_000,
            label="actual statewide sales", color=PALETTE["blue"])
    ax.plot(connected_d["date"], connected_d["forecast_dollars"] / 1_000_000,
            label="county-panel forecast (summed)", color=PALETTE["green"], marker="o", linestyle="--")
    ax.set_ylabel("sales ($ millions)")
    ax.set_xlabel("date")
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.caption(
        f"Pooled CV R² = {diag_d['pooled_r2']:.3f} (± {diag_d['pooled_r2_std']:.3f}) vs. a naive "
        f"\"same county, last year\" guess (no model at all) at R² = {diag_d['naive_r2']:.3f} -- almost "
        f"all of the pooled score is just county size, not real skill. Within-county R² = "
        f"{diag_d['within_r2']:.2f} (± {diag_d['within_r2_std']:.2f}) -- negative, meaning this doesn't "
        f"reliably predict any individual county's ups and downs. Full breakdown on the "
        f"**Investigate & Understand the Data** page."
    )

    st.subheader("County-panel forecast: price per liter ($)")
    actual_p, forecast_p, diag_p = build_county_price_forecast()
    last_point_p = actual_p.iloc[[-1]][["date", "price_per_liter"]].rename(columns={"price_per_liter": "forecast_price_per_liter"})
    connected_p = pd.concat([last_point_p, forecast_p[["date", "forecast_price_per_liter"]]], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_p["date"].tail(36), actual_p["price_per_liter"].tail(36),
            label="actual statewide price per liter", color=PALETTE["blue"])
    ax.plot(connected_p["date"], connected_p["forecast_price_per_liter"],
            label="county-panel forecast (averaged)", color=PALETTE["green"], marker="o", linestyle="--")
    ax.set_ylabel("price per liter ($)")
    ax.set_xlabel("date")
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.caption(
        f"Pooled CV R² = {diag_p['pooled_r2']:.3f} vs. naive R² = {diag_p['naive_r2']:.3f} -- a real "
        f"gap this time, unlike the dollar version above. Within-county R² = {diag_p['within_r2']:.2f} "
        f"(± {diag_p['within_r2_std']:.2f}) -- still negative, so normalizing by volume helped the "
        f"pooled comparison but still didn't produce reliable per-county forecasting skill."
    )

    st.subheader("Metrics summary")
    st.markdown(
        "Every model built across this dashboard, side by side. \"Naive\" is a same-"
        "period-last-year guess with no model at all -- the bar a model has to clear to be "
        "doing real work, not just restating structure that was already obvious."
    )

    sl = state_level_diagnostics()
    forecast_rows = pd.DataFrame([
        {
            "Model": "State-level Lasso (statewide monthly)",
            "Rows": "168",
            "CV R²": f"{sl['r2']:.3f} ± {sl['r2_std']:.3f}",
            "Naive R²": f"{sl['naive_r2']:.3f}",
            "Within-entity R²": "n/a (single series)",
            "Verdict": "Real skill -- best of the three",
        },
        {
            "Model": "County-panel, sales ($)",
            "Rows": "~15,200",
            "CV R²": f"{diag_d['pooled_r2']:.3f} ± {diag_d['pooled_r2_std']:.3f}",
            "Naive R²": f"{diag_d['naive_r2']:.3f}",
            "Within-entity R²": f"{diag_d['within_r2']:.2f} ± {diag_d['within_r2_std']:.2f}",
            "Verdict": "Misleading -- mostly county size, not skill",
        },
        {
            "Model": "County-panel, price per liter ($)",
            "Rows": "~15,200",
            "CV R²": f"{diag_p['pooled_r2']:.3f} ± {diag_p['pooled_r2_std']:.3f}",
            "Naive R²": f"{diag_p['naive_r2']:.3f}",
            "Within-entity R²": f"{diag_p['within_r2']:.2f} ± {diag_p['within_r2_std']:.2f}",
            "Verdict": "Real gap vs. naive, but no per-county skill",
        },
    ])
    st.dataframe(forecast_rows, use_container_width=True, hide_index=True)

    st.subheader("Steps moving forward")
    st.caption(
        "A deliberate change of lens from the rest of this dashboard: the Overview page frames the "
        "goal as helping a public health body *lower* sales. This closing section instead asks the "
        "opposite, business-facing question -- what does the data suggest about *growing* sales -- "
        "as a contrast, not the app's overall stance."
    )
    st.markdown(
        "Reading the associative findings from a retail/revenue angle instead of a harm-reduction "
        "one:\n\n"
        "- **Store access moved the needle historically.** `n_stores` was one of the strongest "
        "features in every model on this dashboard, and the 2023 licensing overhaul (which made it "
        "easier to open/operate a retail license) coincided with the state's largest sales growth "
        "in the dataset. Easier licensing is the single most consistent lever we found.\n"
        "- **Lean into the existing seasonal pattern rather than fight it.** Every year in this "
        "dataset shows the same September-November run-up and a post-holiday drop-off -- inventory, "
        "staffing, and promotions timed to that window would be working with the grain of actual "
        "demand instead of against it.\n"
        "- **The county-panel results above are a real limitation here, not just for harm reduction.** "
        "Negative within-county R² means this data doesn't tell us *which* counties or *which* "
        "months are about to move -- only that store count, price, and season matter in aggregate. "
        "Any county-targeted growth strategy built on this data alone would be guessing past what "
        "the model can actually support.\n"
        "- **Price's effect is genuinely unclear.** It shows up as a real driver in several models here, "
        "but with an inconsistent sign across them -- not a reliable enough signal to base a pricing "
        "strategy on without dedicated experimentation (e.g. an actual price test), which this "
        "wholesale data can't substitute for.\n"
        "- **The clearest next step is better data, not a better model.** Every limitation above "
        "traces back to the same root cause laid out in **Scope & limitations**: this is wholesale "
        "distributor data, not point-of-sale data -- it shows what a *store* bought, not what a "
        "*customer* bought. The county-panel experiment already tested whether simply adding more "
        "rows of this same kind of data would help, and it didn't (within-county R² stayed "
        "negative even with ~15,000 rows). What's actually missing is *local, purchase-level* "
        "signal -- register/POS sales, foot traffic, local promotions, or hyper-local economic "
        "indicators -- data that reflects real consumer purchasing behavior rather than a store's "
        "restocking schedule. Without that, any forecast built on this dataset alone -- to grow "
        "sales or shrink them -- is working from a proxy, not the real signal."
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

render_chrome()

PAGES = {
    "Overview": page_overview,
    "Investigate & Understand the Data": page_investigate,
    "PCA & Multicollinearity": page_pca,
    "What Drives a Sales Dip": page_classification,
    "Forecast": page_forecast,
}

st.sidebar.markdown("### Sections")
page = st.sidebar.radio("Section", list(PAGES.keys()), label_visibility="collapsed")

PAGES[page]()
