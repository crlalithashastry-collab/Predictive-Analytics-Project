# 📈 Predictive Analytics Using Historical Data

## 📌 Project Overview

This project uses historical sales data to build a predictive analytics model that forecasts future sales trends.

The project uses Python and Linear Regression to analyze historical data, evaluate model performance, visualize trends, and predict future sales.

## 🎯 Objectives

- Analyze historical sales data
- Clean and preprocess the dataset
- Build a predictive model
- Evaluate model accuracy
- Forecast future sales
- Visualize historical and predicted trends
- Create an interactive Streamlit dashboard

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## 🤖 Machine Learning Model

The project uses:

**Linear Regression**

The model uses Year as the input feature and Sales as the target variable.

## 📊 Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Results

- MAE: 712.50
- MSE: 522536.04
- RMSE: 722.87
- R² Score: 0.9866

## 🔮 Future Forecast

| Year | Predicted Sales |
|------|----------------:|
| 2025 | 28667.24 |
| 2026 | 30418.10 |
| 2027 | 32168.97 |
| 2028 | 33919.83 |
| 2029 | 35670.69 |

## 📁 Project Structure

```text
Predictive-Analytics-Project/
│
├── data/
│   └── historical_data.csv
│
├── model/
│   └── prediction_model.pkl
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── future_forecast.csv
│   └── future_forecast.png
│
├── predictive_analytics.py
├── app.py
├── requirements.txt
└── README.md