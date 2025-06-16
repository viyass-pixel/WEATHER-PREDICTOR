# WEATHER-PREDICTOR
# Weather Predictor
## AI-Powered Real-Time Weather Forecast Dashboard

This project is a Streamlit-based web application designed to predict current weather conditions and provide 7-day forecasts using a combination of Random Forest and ARIMA models.

---

## Features

- Real-Time Weather Detection  
  Utilizes OpenWeatherMap API and a trained Random Forest Classifier to determine current weather conditions such as rain, sunny, fog, etc.

- Historical Weather Analysis  
  Fetches and visualizes past weather data (Temperature, Humidity, Precipitation) using the Visual Crossing Weather API.

- 7-Day Forecasting  
  Implements the ARIMA model to forecast key weather parameters and uses the Random Forest model to classify expected conditions.

- Trend Visualization  
  Forecasted trends for upcoming temperature, humidity, and precipitation are presented in interactive charts for better interpretation.

---

## Models Used

- Random Forest Classifier  
  Trained on weather features including temperature, humidity, wind speed, precipitation, pressure, UV index, and visibility to classify the weather condition.

- ARIMA (AutoRegressive Integrated Moving Average)  
  A time series forecasting model used to predict future numeric values like temperature, humidity, and precipitation.

---

## Tech Stack

- Python  
- Streamlit  
- scikit-learn  
- statsmodels (ARIMA)  
- OpenWeatherMap API  
- Visual Crossing Weather API  
- Pandas, NumPy, Matplotlib

---

## How to Run the Project

### 1. Clone the Repository

`
git clone https://github.com/your-username/weather-predictor.git
cd weather-predictor
`

### 2. Install Dependencies

`
pip install -r requirements.txt
`

### 3. Launch the Streamlit App

`
streamlit run weather.py
`
