# ================= AQI ANALYSIS & FORECASTING (DEBUGGED – SINGLE BLOCK) =================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# ------------------ LOAD DATA ------------------
# Expected columns: Date, City, AQI, PM2.5, PM10, NO2, SO2 (AQI mandatory)
df = pd.read_csv("aqi_data.csv")

# Date handling
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values(["City", "Date"]).reset_index(drop=True)

# ------------------ CLEANING ------------------
# Interpolate ONLY numeric columns city-wise (safe)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = (
    df.groupby("City")[numeric_cols]
      .apply(lambda x: x.interpolate(method="linear", limit_direction="both"))
      .reset_index(drop=True)
)

# Drop remaining NaNs (edge safety)
df = df.dropna(subset=["AQI"])

# ------------------ CITY SELECTION ------------------
city = "Delhi"
city_df = df[df["City"] == city][["Date", "AQI"]].copy()
city_df = city_df.sort_values("Date").set_index("Date")

# Ensure daily frequency (important for ARIMA/Prophet stability)
city_df = city_df.asfreq("D")
city_df["AQI"] = city_df["AQI"].interpolate(limit_direction="both")

# ------------------ ARIMA ------------------
# Stable order to avoid non-stationary crashes
arima_order = (2, 1, 2)
arima_model = ARIMA(city_df["AQI"], order=arima_order)
arima_fit = arima_model.fit()

# Forecast 180 days
arima_forecast = arima_fit.forecast(steps=180)
arima_index = pd.date_range(
    start=city_df.index[-1] + pd.Timedelta(days=1),
    periods=180,
    freq="D"
)
arima_forecast = pd.Series(arima_forecast.values, index=arima_index)

# Plot ARIMA
plt.figure(figsize=(10, 5))
plt.plot(city_df["AQI"], label="Actual")
plt.plot(arima_forecast, label="ARIMA Forecast")
plt.title(f"ARIMA AQI Forecast – {city}")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------ PROPHET ------------------
prophet_df = city_df.reset_index()
prophet_df.columns = ["ds", "y"]

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)
prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=180, freq="D")
prophet_forecast = prophet_model.predict(future)

# Plot Prophet
prophet_model.plot(prophet_forecast)
plt.title(f"Prophet AQI Forecast – {city}")
plt.tight_layout()
plt.show()

# ------------------ EVALUATION (NO LEAKAGE) ------------------
# Last 30 days as test set
train = city_df.iloc[:-30]
test = city_df.iloc[-30:]

# ARIMA evaluation
arima_eval = ARIMA(train["AQI"], order=arima_order).fit()
arima_pred = arima_eval.forecast(steps=30)

# Prophet evaluation (predict only on test dates)
prophet_train = train.reset_index()
prophet_train.columns = ["ds", "y"]
prophet_test = test.reset_index()

prophet_eval = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)
prophet_eval.fit(prophet_train)
prophet_test_forecast = prophet_eval.predict(prophet_test[["Date"]].rename(columns={"Date": "ds"}))
prophet_pred = prophet_test_forecast["yhat"].values

# Metrics
print("ARIMA MAE:", mean_absolute_error(test["AQI"], arima_pred))
print("Prophet MAE:", mean_absolute_error(test["AQI"], prophet_pred))

# ------------------ HIGH-RISK CITIES ------------------
risk_summary = df.groupby("City")["AQI"].mean().sort_values(ascending=False)
print("\nMost polluted cities (by mean AQI):")
print(risk_summary.head(10))
