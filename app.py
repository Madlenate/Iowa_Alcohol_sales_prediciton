"""
Iowa Liquor Sales -- interactive Streamlit app.

Run with:
    streamlit run app.py

This is a scaffold that ports the core analysis from Project_1_analysis.ipynb
into an interactive dashboard: sales trends, the "what drives sales" linear
regression, and the 3-month-smoothed Lasso forecast model. Edit freely --
it's meant as a starting point, not a finished product.

Uses query_db.q(), so it reads from iowa_liquor.duckdb if present, or falls
back to the parquet files (for the `sales` table only -- the smaller
auxiliary tables like unemployment/homelessness/county_population currently
only exist in the local .duckdb file, so this app needs that file to run
the "What Drives Sales" and "Forecast" pages until those get their own
parquet exports).
"""

import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from query_db import q

st.set_page_config(page_title="Iowa Liquor Sales", layout="wide")


# ---------------------------------------------------------------------------
# Data loading / model fitting (cached so this only re-runs when the
# underlying data changes, not on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_data
def load_statewide():
    """Statewide monthly sales + economic/demographic context. Mirrors cell 37."""
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
def build_forecast_model(df):
    """Baseline residual + smoothed Lasso forecast model. Mirrors cells 39/44/49/51."""
    df = df.copy()
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)

    # Baseline model (n_stores + price + month) -> residual "embedding"
    baseline_cols = ["n_stores", "avg_price_per_bottle"]
    baseline_data = pd.concat([df[baseline_cols], month_dummies], axis=1)
    baseline_model = LinearRegression().fit(baseline_data, df["log_dollars"])
    df["residual"] = df["log_dollars"] - baseline_model.predict(baseline_data)

    # Lag features
    df["residual_lag1"] = df["residual"].shift(1)
    df["n_stores_lag1"] = df["n_stores"].shift(1)
    df["price_lag1"] = df["avg_price_per_bottle"].shift(1)

    # Trailing 3-month smoothing of the target -- only looks backward, so it's safe for forecasting
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
            "log_dollars_smooth_lag12": df["log_dollars_smooth"].iloc[idx_last - 11 + h],
            "residual_lag1": df["residual"].iloc[idx_last],
            "n_stores_lag1": df["n_stores"].iloc[idx_last],
            "price_lag1": df["avg_price_per_bottle"].iloc[idx_last],
        }
        step_X = pd.DataFrame([step_features])[forecast_cols]
        step_dollars = np.expm1(model.predict(step_X)[0])
        rows.append((target_date, step_dollars))
    return pd.DataFrame(rows, columns=["date", "forecast_dollars"])


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("Iowa Liquor Sales")
st.caption("Interactive companion to Project_1_analysis.ipynb -- trends, drivers, and forecasts.")

page = st.sidebar.radio("Section", ["Overview", "What Drives Sales", "Forecast"])

statewide = load_statewide()

if page == "Overview":
    st.header("Statewide sales over time")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total sales (2012-2025)", f"${statewide['total_dollars'].sum() / 1e9:.2f}B")
    col2.metric("Years covered", f"{statewide['sale_year'].min()}-{statewide['sale_year'].max()}")
    col3.metric("Avg monthly sales", f"${statewide['total_dollars'].mean() / 1e6:.1f}M")

    year_min, year_max = int(statewide["sale_year"].min()), int(statewide["sale_year"].max())
    year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))
    filtered = statewide[statewide["sale_year"].between(*year_range)].copy()
    filtered["date"] = pd.to_datetime(dict(year=filtered["sale_year"], month=filtered["month"], day=1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered["date"], filtered["total_dollars"] / 1_000_000, color="#4C72B0")
    ax.set_ylabel("sales ($ millions)")
    ax.set_xlabel("date")
    ax.set_title("Monthly statewide liquor sales")
    st.pyplot(fig)

    st.subheader("Seasonality: average sales by calendar month")
    monthly_avg = statewide.groupby("month")["total_dollars"].mean() / 1_000_000
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar([calendar.month_abbr[m] for m in monthly_avg.index], monthly_avg.values, color="#4C72B0")
    ax2.set_ylabel("avg sales ($ millions)")
    st.pyplot(fig2)
    st.caption("Sept-Nov stands out -- likely tied to college football season.")

elif page == "What Drives Sales":
    st.header("What drives month-to-month sales")
    st.write(
        "A linear regression of log(sales) on the features you pick below -- "
        "same idea as cell 32's model comparison in the notebook."
    )

    feature_choices = st.multiselect(
        "Features to include",
        ["n_stores", "avg_price_per_bottle", "unemployment_rate", "overall_homeless",
         "unsheltered_homeless", "stores_per_10k_capita"],
        default=["n_stores", "avg_price_per_bottle"],
    )

    if feature_choices:
        data = statewide.dropna(subset=feature_choices + ["log_dollars"])
        X = data[feature_choices]
        y = data["log_dollars"]
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

        st.metric("R^2", f"{r2:.3f}")
        coefs = pd.Series(model.coef_, index=feature_choices).sort_values(key=abs, ascending=False)
        st.bar_chart(coefs)
    else:
        st.info("Pick at least one feature.")

else:  # Forecast
    st.header("Forecasting next month's sales (Lasso)")
    st.write(
        "Mirrors the notebook's final forecast model: a Lasso regression on a "
        "3-month-smoothed sales target, using last year's level plus recent "
        "store/price/momentum signals. The full model comparison behind this "
        "choice (Ridge/ElasticNet/Random Forest/XGBoost/SARIMAX/naive baseline) "
        "lives in the notebook, not here."
    )

    df_feat, model, forecast_cols, fc_data, X_forecast = build_forecast_model(statewide)

    horizons = st.slider("Months to forecast ahead", 1, 6, 3)
    forecast_df = multi_month_forecast(df_feat, model, forecast_cols, horizons=horizons)

    plot_hist = df_feat.loc[fc_data.index].copy()
    plot_hist["date"] = pd.to_datetime(dict(year=plot_hist["sale_year"], month=plot_hist["month"], day=1))
    plot_hist["forecast_date"] = plot_hist["date"] + pd.DateOffset(months=1)
    plot_hist["forecast_dollars"] = np.expm1(model.predict(X_forecast))
    plot_hist = plot_hist.sort_values("forecast_date")

    full_actual = df_feat.dropna(subset=["log_dollars_smooth"]).copy()
    full_actual["date"] = pd.to_datetime(dict(year=full_actual["sale_year"], month=full_actual["month"], day=1))
    full_actual["actual_smooth_dollars"] = np.expm1(full_actual["log_dollars_smooth"])

    last_hist_point = pd.DataFrame({
        "date": [plot_hist["forecast_date"].iloc[-1]],
        "forecast_dollars": [plot_hist["forecast_dollars"].iloc[-1]],
    })
    forecast_connected = pd.concat([last_hist_point, forecast_df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_actual["date"], full_actual["actual_smooth_dollars"] / 1_000_000,
            label="actual (3-month smoothed)", color="#4C72B0")
    ax.plot(plot_hist["forecast_date"], plot_hist["forecast_dollars"] / 1_000_000,
            label="1-month-ahead forecast (history)", color="#C44E52", linestyle="--", alpha=0.7)
    ax.plot(forecast_connected["date"], forecast_connected["forecast_dollars"] / 1_000_000,
            label="forward forecast", color="#55A868", marker="o", linestyle="--")
    ax.set_ylabel("sales ($ millions)")
    ax.set_xlabel("date")
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.subheader("Forecast values")
    st.dataframe(forecast_df.style.format({"forecast_dollars": "${:,.0f}"}))

    st.caption(
        "Months beyond the first carry forward the last known store count, price, and "
        "momentum -- only the year-ago seasonal anchor updates with real data at every "
        "horizon, so treat anything past month 1 as lower-confidence."
    )
