import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import numpy as np
import pickle
import warnings
from statsmodels.tsa.arima.model import ARIMA

# -------------------- Load ML Model --------------------
with open("weather_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# -------------------- API Keys --------------------
VC_API_KEY = "DMTU44FSNY7KT4XWE9PZAXD2H"  # Visual Crossing
OWM_API_KEY = "0f5b7006422b107bfefb64b1336a19e8"  # OpenWeatherMap

# -------------------- Utility Functions --------------------
def get_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "Temperature": data["main"]["temp"],
            "Humidity": data["main"]["humidity"],
            "Wind Speed": data["wind"]["speed"] * 3.6,
            "Pressure": data["main"]["pressure"],
            "Precipitation": data.get("rain", {}).get("1h", 0),
            "UV Index": 5,
            "Visibility": data["visibility"] / 1000
        }
    else:
        return None

def predict_weather(features):
    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)
    encoded_prediction = model.predict(input_scaled)[0]
    return label_encoder.inverse_transform([encoded_prediction])[0]

def get_historical_weather(city, start_date, end_date):
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{city}/{start_date}/{end_date}?unitGroup=metric&include=days&key={VC_API_KEY}&contentType=json"
    )
    response = requests.get(url)
    if response.status_code == 200:
        days = response.json()["days"]
        df = pd.DataFrame(days)[["datetime", "temp", "humidity", "windspeed", "precip"]]
        df.columns = ["Date", "Temperature", "Humidity", "Wind Speed", "Precipitation"]
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    else:
        return None

# -------------------- Streamlit App --------------------
st.set_page_config(layout="wide")
st.title("🌤️ AI Weather Predictor & 7-Day Forecast")

city = st.text_input("Enter City Name:", "Nagercoil")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📅 Start Date", date.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("📅 End Date", date.today())

# 🌍 Real-Time Weather Prediction
if st.button("🌍 Get Current Weather & Predict"):
    weather_data = get_weather_data(city)
    if weather_data:
        st.subheader(f"🌡️ Real-Time Weather Data for {city}")
        st.json(weather_data)

        predicted = predict_weather([
            weather_data["Temperature"],
            weather_data["Humidity"],
            weather_data["Wind Speed"],
            weather_data["Precipitation"],
            weather_data["Pressure"],
            weather_data["UV Index"],
            weather_data["Visibility"]
        ])
        st.success(f"🌈 Predicted Weather: **{predicted}**")
    else:
        st.error("⚠️ Failed to fetch current weather.")

# 📈 Historical Weather Trend
if st.button("📈 Show Historical Trend"):
    df = get_historical_weather(city, start_date, end_date)
    if df is not None:
        st.subheader(f"📊 Weather Trend in {city} ({start_date} to {end_date})")
        st.line_chart(df.set_index("Date")[["Temperature", "Humidity", "Precipitation"]])
    else:
        st.error("⚠️ Could not fetch historical data.")

# 🔮 Forecast 7-Day Weather Condition
if st.button("🔮 Forecast Next 7 Days Weather Condition"):
    df = get_historical_weather(city, start_date, end_date)
    if df is not None and len(df) >= 10:
        warnings.filterwarnings("ignore")
        df.set_index("Date", inplace=True)

        forecasts = {}
        for feature in ["Temperature", "Humidity", "Wind Speed", "Precipitation"]:
            model_arima = ARIMA(df[feature], order=(5, 1, 0)).fit()
            forecasts[feature] = model_arima.forecast(steps=7)

        pressure_avg = 1013
        uv_index = 5
        visibility = 8
        predicted_weather = []
        forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=7)

        for i in range(7):
            feature_vector = [
                forecasts["Temperature"][i],
                forecasts["Humidity"][i],
                forecasts["Wind Speed"][i],
                forecasts["Precipitation"][i],
                pressure_avg,
                uv_index,
                visibility
            ]
            pred = predict_weather(feature_vector)
            predicted_weather.append(pred)

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted Weather": predicted_weather
        })

        st.subheader("📅 7-Day Weather Condition Forecast")
        st.dataframe(forecast_df, use_container_width=True)
    else:
        st.warning("⚠️ Need at least 10 days of historical data for reliable forecasting.")
