import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Create required folders
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Historical sales dataset
data = {
    "Year": [
        2015, 2016, 2017, 2018, 2019,
        2020, 2021, 2022, 2023, 2024
    ],
    "Sales": [
        12000, 13500, 15000, 16800, 18000,
        17500, 21000, 23500, 26000, 28500
    ]
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("data/historical_data.csv", index=False)

print("\n========== HISTORICAL DATA ==========")
print(df)

# Features and target
X = df[["Year"]]
y = df["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save trained model
joblib.dump(model, "model/prediction_model.pkl")

# Actual vs Predicted graph
plt.figure(figsize=(10, 6))

plt.scatter(
    X_test,
    y_test,
    label="Actual"
)

plt.scatter(
    X_test,
    y_pred,
    label="Predicted"
)

plt.xlabel("Year")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.grid(True)

plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# Future years
future_years = pd.DataFrame({
    "Year": [2025, 2026, 2027, 2028, 2029]
})

# Predict future sales
future_predictions = model.predict(future_years)

forecast = pd.DataFrame({
    "Year": future_years["Year"],
    "Predicted_Sales": future_predictions.round(2)
})

print("\n========== FUTURE SALES FORECAST ==========")
print(forecast)

# Save future forecast
forecast.to_csv(
    "outputs/future_forecast.csv",
    index=False
)

# Future forecast graph
plt.figure(figsize=(10, 6))

plt.plot(
    df["Year"],
    df["Sales"],
    marker="o",
    label="Historical Sales"
)

plt.plot(
    future_years["Year"],
    future_predictions,
    marker="o",
    linestyle="--",
    label="Future Forecast"
)

plt.xlabel("Year")
plt.ylabel("Sales")
plt.title("Historical Sales and Future Forecast")
plt.legend()
plt.grid(True)

plt.savefig("outputs/future_forecast.png")
plt.close()

print("\n======================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("======================================")