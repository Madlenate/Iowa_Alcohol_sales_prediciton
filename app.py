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
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def page_forecast():
    st.header("Forecast")
    st.info(
        "\U0001F6A7 **Under revision.** The forecasting approach (Lasso on a 3-month-smoothed "
        "target, using last year's level plus recent store/price/momentum signals) is being "
        "reworked, so this section is intentionally left blank for now.\n\n"
        "The lag-alignment bug that broke the previous version's seasonal anchor is already "
        "fixed in `build_forecast_model` / `multi_month_forecast` in app.py -- they're just not "
        "wired into this page yet."
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
