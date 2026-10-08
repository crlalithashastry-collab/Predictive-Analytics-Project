import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("📈 Predictive Analytics Dashboard")
st.write("Historical Sales Analysis and Future Sales Forecasting")

# ==========================================
# LOAD DATA AND MODEL
# ==========================================

data = pd.read_csv("data/historical_data.csv")
forecast = pd.read_csv("outputs/future_forecast.csv")
model = joblib.load("model/prediction_model.pkl")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("📊 Project Information")

st.sidebar.write("**Model:** Linear Regression")
st.sidebar.write("**Dataset:** Historical Sales")
st.sidebar.write("**Forecast Period:** 2025–2029")

# ==========================================
# KEY METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest Sales",
        f"{data['Sales'].iloc[-1]:,.0f}"
    )

with col2:
    st.metric(
        "Forecast 2025",
        f"{forecast['Predicted_Sales'].iloc[0]:,.0f}"
    )

with col3:
    st.metric(
        "Forecast 2029",
        f"{forecast['Predicted_Sales'].iloc[-1]:,.0f}"
    )

st.divider()

# ==========================================
# HISTORICAL DATA
# ==========================================

st.subheader("📋 Historical Sales Data")

st.dataframe(
    data,
    width="stretch"
)

# ==========================================
# HISTORICAL SALES CHART
# ==========================================

st.subheader("📊 Historical Sales Trend")

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(
    data["Year"],
    data["Sales"],
    marker="o"
)

ax1.set_xlabel("Year")
ax1.set_ylabel("Sales")
ax1.set_title("Historical Sales Trend")
ax1.grid(True)

st.pyplot(fig1)

# ==========================================
# FUTURE FORECAST
# ==========================================

st.subheader("🔮 Future Sales Forecast")

st.dataframe(
    forecast,
    width="stretch"
)

# ==========================================
# FORECAST CHART
# ==========================================

st.subheader("📈 Historical Sales + Future Forecast")

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(
    data["Year"],
    data["Sales"],
    marker="o",
    label="Historical Sales"
)

ax2.plot(
    forecast["Year"],
    forecast["Predicted_Sales"],
    marker="o",
    linestyle="--",
    label="Future Forecast"
)

ax2.set_xlabel("Year")
ax2.set_ylabel("Sales")
ax2.set_title("Historical Sales and Future Forecast")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# ==========================================
# PREDICT SALES FOR FUTURE YEAR
# ==========================================

st.subheader("🎯 Predict Sales for a Future Year")

year = st.number_input(
    "Enter a year",
    min_value=2025,
    max_value=2050,
    value=2025,
    step=1
)

if st.button("Predict Sales"):

    prediction = model.predict(
        pd.DataFrame({"Year": [year]})
    )

    st.success(
        f"Predicted sales for {year}: "
        f"{prediction[0]:,.2f}"
    )

# ==========================================
# PROJECT SUMMARY
# ==========================================

st.divider()

st.subheader("📝 Project Summary")

st.write(
    """
    This project uses historical sales data to build a predictive
    analytics model using Linear Regression.

    The model analyzes historical sales trends and forecasts
    future sales values.

    The project includes:

    • Data collection and preprocessing
    • Linear Regression model
    • Model training
    • Model evaluation
    • Future sales forecasting
    • Data visualization
    • Interactive Streamlit dashboard
    """
)

# ==========================================
# SUCCESS MESSAGE
# ==========================================

st.success(
    "✅ Predictive Analytics Project is working successfully!"
)