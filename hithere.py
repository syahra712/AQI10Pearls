import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import streamlit as st

# ============================================================
#  STEP 1 — FETCH WEATHER & POLLUTANT DATA
# ============================================================
def fetch_monthly_data(latitude, longitude, start_date, end_date):
    """Fetch one month of weather and pollutant data from Open-Meteo APIs."""
    # --- WEATHER DATA ---
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "surface_pressure", "cloud_cover", "windspeed_10m", "winddirection_10m"
        ],
        "timezone": "auto"
    }

    w_resp = requests.get(weather_url, params=weather_params)
    w_resp.raise_for_status()
    w_data = w_resp.json()

    weather_df = pd.DataFrame({
        "datetime": w_data["hourly"]["time"],
        "temperature_2m": w_data["hourly"]["temperature_2m"],
        "relative_humidity_2m": w_data["hourly"]["relative_humidity_2m"],
        "dew_point_2m": w_data["hourly"]["dew_point_2m"],
        "apparent_temperature": w_data["hourly"]["apparent_temperature"],
        "precipitation": w_data["hourly"]["precipitation"],
        "rain": w_data["hourly"]["rain"],
        "snowfall": w_data["hourly"]["snowfall"],
        "surface_pressure": w_data["hourly"]["surface_pressure"],
        "cloud_cover": w_data["hourly"]["cloud_cover"],
        "windspeed_10m": w_data["hourly"]["windspeed_10m"],
        "winddirection_10m": w_data["hourly"]["winddirection_10m"]
    })
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    # --- AIR QUALITY DATA ---
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone",
            "aerosol_optical_depth", "dust", "uv_index"
        ],
        "timezone": "auto"
    }

    aq_resp = requests.get(aq_url, params=aq_params)
    aq_resp.raise_for_status()
    aq_data = aq_resp.json()

    aq_df = pd.DataFrame({
        "datetime": aq_data["hourly"]["time"],
        "pm10": aq_data["hourly"]["pm10"],
        "pm2_5": aq_data["hourly"]["pm2_5"],
        "carbon_monoxide": aq_data["hourly"]["carbon_monoxide"],
        "nitrogen_dioxide": aq_data["hourly"]["nitrogen_dioxide"],
        "sulphur_dioxide": aq_data["hourly"]["sulphur_dioxide"],
        "ozone": aq_data["hourly"]["ozone"],
        "aerosol_optical_depth": aq_data["hourly"]["aerosol_optical_depth"],
        "dust": aq_data["hourly"]["dust"],
        "uv_index": aq_data["hourly"]["uv_index"]
    })
    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"])

    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")
    return merged


def fetch_yearly_data(latitude, longitude, start_year, end_year):
    """Fetch data month-by-month for a full year."""
    all_data = []
    start = datetime(start_year, 10, 21)
    end = datetime(end_year, 10, 21)
    current = start

    while current < end:
        month_end = (current + timedelta(days=30))
        if month_end > end:
            month_end = end
        print(f"Fetching data from {current.date()} to {month_end.date()}...")
        df = fetch_monthly_data(latitude, longitude, current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
        all_data.append(df)
        current = month_end + timedelta(days=1)

    return pd.concat(all_data, ignore_index=True)


# ============================================================
#  STEP 2 — COMPUTE FEATURES & STORE IN HOPSWORKS
# ============================================================
def create_feature_group(df):
    """Compute derived features and push to Hopsworks Feature Store."""
    project = hopsworks.login()
    fs = project.get_feature_store()

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["aqi_estimated"] = df["pm2_5"] * 0.5 + df["pm10"] * 0.3 + df["ozone"] * 0.2
    df["aqi_change_rate"] = df["aqi_estimated"].diff().fillna(0)

    feature_group = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["datetime"],
        description="Weather + AQI features with time and change rate"
    )

    feature_group.insert(df)
    print("✅ Features stored in Hopsworks Feature Store!")

    return df


# ============================================================
#  STEP 3 — TRAIN AND REGISTER MODEL
# ============================================================
def train_and_register_model(df):
    """Train RandomForest on features and register model in Hopsworks."""
    project = hopsworks.login()
    mr = project.get_model_registry()

    df = df.dropna()
    X = df[["temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "windspeed_10m", "surface_pressure", "cloud_cover",
            "pm2_5", "pm10", "ozone", "aqi_change_rate", "hour", "month"]]
    y = df["aqi_estimated"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"✅ Model trained. RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Create directory if it doesn't exist (FIX)
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/rf_model.pkl")

    model_meta = mr.python.create_model(
        name="aqi_predictor_rf",
        metrics={"rmse": rmse, "r2": r2},
        description="Random Forest model for AQI prediction"
    )
    model_meta.save(f"{model_dir}/rf_model.pkl")
    print("✅ Model registered in Hopsworks!")

    return model


# ============================================================
#  STEP 4 — STREAMLIT DASHBOARD
# ============================================================
def run_dashboard(model, df):
    st.title("🌫️ Karachi AQI Prediction Dashboard")

    latest_data = df.sort_values("datetime").tail(72)
    X = latest_data[["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                     "windspeed_10m", "surface_pressure", "cloud_cover",
                     "pm2_5", "pm10", "ozone", "aqi_change_rate", "hour", "month"]]
    preds = model.predict(X)

    result_df = pd.DataFrame({
        "datetime": latest_data["datetime"],
        "Predicted AQI": preds
    }).set_index("datetime")

    st.line_chart(result_df)
    st.metric("🌤 Latest AQI Prediction", round(preds[-1], 2))
    st.write("Data Source: Open-Meteo APIs | Model: Random Forest | Powered by Hopsworks")


# ============================================================
#  MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    latitude, longitude = 24.8607, 67.0011
    print("🌍 Fetching 1 year of data for Karachi...")
    df = fetch_yearly_data(latitude, longitude, 2024, 2025)
    print(f"✅ Total records fetched: {len(df)}")

    # Save locally for backup
    df.to_csv("karachi_weather_pollutants.csv", index=False)

    # Store features & train model
    df = create_feature_group(df)
    model = train_and_register_model(df)

    # Launch dashboard
    run_dashboard(model, df)
