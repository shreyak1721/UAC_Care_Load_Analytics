import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UAC System Capacity & Care Load Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("System Capacity & Care Load Analytics")
st.subheader("Unaccompanied Children (UAC) Care System")

st.markdown(
    """
    This dashboard analyzes daily operational data across the
    CBP custody and HHS care pipeline.

    The analysis covers system load, transfers, discharges,
    care-load pressure, volatility, and time-series forecasting.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    possible_files = [
        "HHS_Unaccompanied_Alien_Children_Program.csv",
        "HHS_Unaccompanied_Alien_Children_Program (1).csv"
    ]

    file_path = None

    for file in possible_files:
        try:
            pd.read_csv(file)
            file_path = file
            break
        except:
            continue

    if file_path is None:
        raise FileNotFoundError(
            "Dataset CSV not found. Please place the CSV "
            "in the same folder as app.py."
        )

    df = pd.read_csv(file_path)

    return df


df = load_data()


# ============================================================
# DATA CLEANING
# ============================================================

df = df.dropna(how="all").copy()

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

hhs_column = "Children in HHS Care"

df[hhs_column] = (
    df[hhs_column]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df[hhs_column] = pd.to_numeric(
    df[hhs_column],
    errors="coerce"
)

numeric_columns = [
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = (
    df
    .dropna(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# DATA QUALITY
# ============================================================

duplicate_dates = df["Date"].duplicated().sum()

missing_values = df.isnull().sum().sum()

transfer_violations = (
    df["Children transferred out of CBP custody"]
    > df["Children in CBP custody"]
).sum()

discharge_violations = (
    df["Children discharged from HHS Care"]
    > df["Children in HHS Care"]
).sum()


# ============================================================
# DERIVED METRICS
# ============================================================

df["Total System Load"] = (
    df["Children in CBP custody"]
    + df["Children in HHS Care"]
)

df["Net Intake Pressure"] = (
    df["Children transferred out of CBP custody"]
    - df["Children discharged from HHS Care"]
)

df["Care Load Growth Rate"] = (
    df["Total System Load"]
    .pct_change()
    * 100
)

df["7-Day Rolling Load"] = (
    df["Total System Load"]
    .rolling(7)
    .mean()
)

df["14-Day Rolling Load"] = (
    df["Total System Load"]
    .rolling(14)
    .mean()
)

df["Cumulative Net Intake"] = (
    df["Net Intake Pressure"]
    .cumsum()
)

df["Discharge Offset Ratio"] = np.where(
    df["Children transferred out of CBP custody"] > 0,
    df["Children discharged from HHS Care"]
    / df["Children transferred out of CBP custody"],
    np.nan
)


# ============================================================
# PRESSURE FLAGS
# ============================================================

load_75 = df["Total System Load"].quantile(0.75)
load_90 = df["Total System Load"].quantile(0.90)

df["High Load"] = (
    df["Total System Load"] >= load_75
)

df["Very High Load"] = (
    df["Total System Load"] >= load_90
)

df["Sustained High Load"] = (
    df["7-Day Rolling Load"] >= load_75
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Dashboard Filters")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_dates) == 2:

    start_date = pd.Timestamp(selected_dates[0])
    end_date = pd.Timestamp(selected_dates[1])

    filtered_df = df[
        (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ].copy()

else:
    filtered_df = df.copy()


# ============================================================
# OVERVIEW KPIs
# ============================================================

st.header("System Load Overview")

latest = df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Total System Load",
        f"{latest['Total System Load']:,.0f}"
    )

with col2:
    st.metric(
        "CBP Custody",
        f"{latest['Children in CBP custody']:,.0f}"
    )

with col3:
    st.metric(
        "HHS Care",
        f"{latest['Children in HHS Care']:,.0f}"
    )

with col4:
    st.metric(
        "Net Intake Pressure",
        f"{latest['Net Intake Pressure']:,.0f}"
    )


# ============================================================
# LOAD TREND
# ============================================================

st.header("System Load Trend")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Total System Load"],
        mode="lines",
        name="Total System Load"
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["7-Day Rolling Load"],
        mode="lines",
        name="7-Day Rolling Average"
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["14-Day Rolling Load"],
        mode="lines",
        name="14-Day Rolling Average"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Children Under Care",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# CBP VS HHS
# ============================================================

st.header("CBP vs HHS Care Load")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Children in CBP custody"],
        mode="lines",
        name="CBP Custody"
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Children in HHS Care"],
        mode="lines",
        name="HHS Care"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Number of Children",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# TRANSFERS VS DISCHARGES
# ============================================================

st.header("Transfers and Discharges")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Children transferred out of CBP custody"],
        mode="lines",
        name="Transfers into HHS"
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Children discharged from HHS Care"],
        mode="lines",
        name="HHS Discharges"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Children",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# NET INTAKE PRESSURE
# ============================================================

st.header("Net Intake Pressure")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Net Intake Pressure"],
        mode="lines",
        name="Net Intake Pressure"
    )
)

fig.add_hline(
    y=0,
    line_width=1
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Transfers − Discharges",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# BACKLOG INDICATOR
# ============================================================

st.header("Cumulative Net Intake")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Cumulative Net Intake"],
        mode="lines",
        name="Cumulative Net Intake"
    )
)

fig.add_hline(
    y=0,
    line_width=1
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Cumulative Transfers − Discharges",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# PRESSURE ANALYSIS
# ============================================================

st.header("Capacity Pressure Analysis")

pressure_col1, pressure_col2, pressure_col3 = st.columns(3)

with pressure_col1:
    st.metric(
        "75th Percentile Load",
        f"{load_75:,.0f}"
    )

with pressure_col2:
    st.metric(
        "90th Percentile Load",
        f"{load_90:,.0f}"
    )

with pressure_col3:
    st.metric(
        "High-Load Days",
        f"{df['High Load'].sum():,.0f}"
    )


pressure_days = df[
    df["Sustained High Load"]
][
    [
        "Date",
        "Total System Load",
        "7-Day Rolling Load",
        "Net Intake Pressure"
    ]
].sort_values(
    "Date",
    ascending=False
)

st.subheader("Sustained High-Load Observations")

st.dataframe(
    pressure_days,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VOLATILITY
# ============================================================

st.header("Care Load Volatility")

mean_load = df["Total System Load"].mean()
std_load = df["Total System Load"].std()

volatility_index = std_load / mean_load

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average System Load",
        f"{mean_load:,.0f}"
    )

with col2:
    st.metric(
        "Standard Deviation",
        f"{std_load:,.0f}"
    )

with col3:
    st.metric(
        "Volatility Index",
        f"{volatility_index:.4f}"
    )


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

st.header("Monthly System Load")

df["Month"] = df["Date"].dt.to_period("M").astype(str)

monthly = (
    df.groupby("Month")
    .agg(
        Average_System_Load=(
            "Total System Load",
            "mean"
        ),
        Maximum_System_Load=(
            "Total System Load",
            "max"
        ),
        Average_Net_Intake=(
            "Net Intake Pressure",
            "mean"
        ),
        Total_Transfers=(
            "Children transferred out of CBP custody",
            "sum"
        ),
        Total_Discharges=(
            "Children discharged from HHS Care",
            "sum"
        )
    )
    .reset_index()
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=monthly["Month"],
        y=monthly["Average_System_Load"],
        name="Average System Load"
    )
)

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Average System Load"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# FORECAST
# ============================================================

st.header("30-Observation System Load Forecast")

forecast_series = (
    df["Total System Load"]
    .dropna()
    .reset_index(drop=True)
)

forecast_available = len(forecast_series) >= 60

if forecast_available:

    split_index = int(len(forecast_series) * 0.80)

    train = forecast_series.iloc[:split_index]
    test = forecast_series.iloc[split_index:]

    try:

        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        result = model.fit(disp=False)

        predictions = result.forecast(
            steps=len(test)
        )

        mae = mean_absolute_error(
            test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                predictions
            )
        )

        # Final model using all available observations
        final_model = SARIMAX(
            forecast_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        final_result = final_model.fit(
            disp=False
        )

        future = final_result.get_forecast(
            steps=30
        )

        future_mean = future.predicted_mean
        future_ci = future.conf_int()

        future_dates = pd.date_range(
            start=df["Date"].max()
            + pd.Timedelta(days=1),
            periods=30,
            freq="D"
        )

        forecast_output = pd.DataFrame({
            "Date": future_dates,
            "Forecast": future_mean.values,
            "Lower Bound": future_ci.iloc[:, 0].values,
            "Upper Bound": future_ci.iloc[:, 1].values
        })

        # Metrics
        fc1, fc2 = st.columns(2)

        with fc1:
            st.metric(
                "Test MAE",
                f"{mae:,.2f}"
            )

        with fc2:
            st.metric(
                "Test RMSE",
                f"{rmse:,.2f}"
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Total System Load"],
                mode="lines",
                name="Historical"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_output["Date"],
                y=forecast_output["Forecast"],
                mode="lines",
                name="Forecast"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_output["Date"],
                y=forecast_output["Upper Bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_output["Date"],
                y=forecast_output["Lower Bound"],
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name="Confidence Interval"
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Total System Load",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Forecast Table")

        st.dataframe(
            forecast_output,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.warning(
            f"Forecast could not be generated: {e}"
        )

else:

    st.warning(
        "Not enough observations available for forecasting."
    )


# ============================================================
# DATA QUALITY
# ============================================================

st.header("Data Quality & Validation")

dq1, dq2, dq3, dq4 = st.columns(4)

with dq1:
    st.metric(
        "Records",
        f"{len(df):,}"
    )

with dq2:
    st.metric(
        "Duplicate Dates",
        f"{duplicate_dates:,}"
    )

with dq3:
    st.metric(
        "Missing Values",
        f"{missing_values:,}"
    )

with dq4:
    st.metric(
        "Transfer Violations",
        f"{transfer_violations:,}"
    )

st.write(
    "Discharge > HHS Care violations:",
    discharge_violations
)

st.subheader("Missing Values by Column")

missing_table = (
    df[numeric_columns]
    .isnull()
    .sum()
    .reset_index()
)

missing_table.columns = [
    "Column",
    "Missing Values"
]

st.dataframe(
    missing_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RAW DATA
# ============================================================

with st.expander("View Analytical Dataset"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "System Capacity & Care Load Analytics for "
    "Unaccompanied Children | Analytical dashboard"
)