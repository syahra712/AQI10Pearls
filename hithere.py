#!/usr/bin/env python3
import os
import time
import logging
import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta, timezone
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# streamlit import left but only used when running dashboard mode
try:
    import streamlit as st
except Exception:
    st = None

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Silence noisy logs if present
logging.getLogger("rdkafka").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aqi")

# ---------- CONFIG ----------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqipredictionn")
FEATURE_GROUP_NAME = "aqi_features"
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
FEATURE_GROUP_VERSION = 1
MODEL_DIR = "./model"
CSV_PATH = "features_latest.csv"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------- HTTP session with retries ----------
def make_session(total_retries=5, backoff=1.0):
    session = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ============================================================
#  STEP 1 — FETCH WEATHER & POLLUTANT DATA
# ============================================================
def fetch_monthly_data(latitude, longitude, start_date, end_date, session=None):
    """Fetch weather + air quality hourly data for the given date range using Open-Meteo archive APIs."""
    if session is None:
        session = make_session()

    def safe_get(url, params):
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    logger.info("Fetching weather for %s → %s", start_date, end_date)
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "surface_pressure", "cloud_cover", "windspeed_10m", "winddirection_10m"
        ]),
        "timezone": "UTC"
    }
    w_data = safe_get(weather_url, weather_params)
    weather_hourly = w_data.get("hourly", {})

    logger.info("Fetching air quality for %s → %s", start_date, end_date)
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone",
            "aerosol_optical_depth", "dust", "uv_index"
        ]),
        "timezone": "UTC"
    }
    aq_data = safe_get(aq_url, aq_params)
    aq_hourly = aq_data.get("hourly", {})

    # Build DataFrames
    if "time" not in weather_hourly or "time" not in aq_hourly:
        logger.warning("No hourly time data returned for the period.")
        return pd.DataFrame()

    weather_df = pd.DataFrame(weather_hourly).rename(columns={"time": "datetime"})
    aq_df = pd.DataFrame(aq_hourly).rename(columns={"time": "datetime"})
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], utc=True)
    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"], utc=True)

    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")
    return merged


def fetch_range_data(latitude, longitude, days=180):
    """Fetch last ~6 months of data (default 180 days) in 7-day chunks."""
    session = make_session()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    current = start
    chunks = []
    while current < end:
        chunk_end = min(end, current + timedelta(days=7))
        s = current.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        try:
            df_chunk = fetch_monthly_data(latitude, longitude, s, e, session=session)
            if not df_chunk.empty:
                chunks.append(df_chunk)
        except Exception as exc:
            logger.warning("Fetch failed for %s to %s: %s", s, e, exc)
        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


# ============================================================
#  STEP 2 — COMPUTE FEATURES & STORE IN HOPSWORKS
# ============================================================
def create_feature_group(df, feature_group_name=FEATURE_GROUP_NAME):
    """Compute derived features and push to Hopsworks Feature Store."""
    if df.empty:
        raise ValueError("Empty dataframe passed to create_feature_group")

    logger.info("Preparing features (time features + derived AQI)...")
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["aqi_estimated"] = df.get("pm2_5", 0) * 0.5 + df.get("pm10", 0) * 0.3 + df.get("ozone", 0) * 0.2
    df["aqi_change_rate"] = df["aqi_estimated"].diff().fillna(0)

    logger.info("Logging into Hopsworks...")
    project = hopsworks.login(     api_key_value=HOPSWORKS_API_KEY,     project=HOPSWORKS_PROJECT,     host=HOPSWORKS_HOST )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=FEATURE_GROUP_VERSION,
        primary_key=["datetime"],
        description="Weather + AQI features with time and change rate"
    )

    fg.insert(df, write_options={"wait_for_job": False})
    logger.info("Features pushed to Hopsworks feature store.")
    return df


# ============================================================
#  STEP 3 — TRAIN AND REGISTER MODEL
# ============================================================
def wait_for_materialization(job_name="aqi_features_1_offline_fg_materialization", timeout=300):
    try:
        project = hopsworks.login(     api_key_value=HOPSWORKS_API_KEY,     project=HOPSWORKS_PROJECT,     host=HOPSWORKS_HOST )
        job = project.get_jobs().get_job(job_name)
        if job:
            logger.info("Waiting for materialization job %s...", job_name)
            job.await_completion(timeout_sec=timeout)
            logger.info("Materialization finished.")
    except Exception as e:
        logger.warning("Could not check materialization job: %s", e)


def train_and_register_model(df):
    """Train RandomForest on full dataset and register model in Hopsworks."""
    if df.empty:
        raise ValueError("Empty dataframe passed to training.")

    wait_for_materialization()
    project = hopsworks.login(     api_key_value=HOPSWORKS_API_KEY,     project=HOPSWORKS_PROJECT,     host=HOPSWORKS_HOST )
    mr = project.get_model_registry()
    df = df.dropna()

    features = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "windspeed_10m", "surface_pressure", "cloud_cover",
                "pm2_5", "pm10", "ozone", "aqi_change_rate", "hour", "month"]
    X = df[features]
    y = df["aqi_estimated"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    logger.info("✅ Model trained. RMSE: %.3f, R²: %.3f", rmse, r2)

    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    joblib.dump(model, local_path)

    try:
        model_meta = mr.python.create_model(
            name="aqi_predictor_rf",
            metrics={"rmse": float(rmse), "r2": float(r2)},
            description="Random Forest model for AQI prediction"
        )
        model_meta.save(local_path)
        logger.info("Model registered in Hopsworks registry.")
    except Exception as e:
        logger.warning("Model registration failed: %s", e)

    return model


# ============================================================
#  MAIN MODES
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="AQI pipeline runner")
    p.add_argument("--mode", choices=["backfill", "fetch", "train", "dashboard", "run-once"], default="run-once")
    p.add_argument("--days", type=int, default=180, help="Days to backfill/fetch (default 180 = 6 months)")
    p.add_argument("--hours", type=int, default=24, help="Hours to fetch (default 24)")
    p.add_argument("--lat", type=float, default=24.8607)
    p.add_argument("--lon", type=float, default=67.0011)
    return p.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    if mode == "backfill":
        logger.info("Backfilling last %d days (~6 months)...", args.days)
        df = fetch_range_data(args.lat, args.lon, days=args.days)
        df.to_csv(CSV_PATH, index=False)
        create_feature_group(df)
        logger.info("Backfill complete with %d rows.", len(df))
        return

    if mode == "fetch":
        logger.info("Fetching new hourly data...")
        df = fetch_range_data(args.lat, args.lon, days=max(1, args.hours // 24))
        if os.path.exists(CSV_PATH):
            old = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
            combined = pd.concat([old, df]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
        else:
            combined = df
        combined.to_csv(CSV_PATH, index=False)
        create_feature_group(df)
        logger.info("Appended %d new rows (total %d).", len(df), len(combined))
        return

    if mode == "train":
        if not os.path.exists(CSV_PATH):
            logger.error("features_latest.csv not found. Run backfill first.")
            return
        df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
        train_and_register_model(df)
        logger.info("Model retrained on %d rows (%.1f months).", len(df), len(df) / 720)
        return

    if mode == "dashboard":
        model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
        df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
        run_dashboard(model, df)
        return

    if mode == "run-once":
        logger.info("Running one-shot mode: fetch → append → train")
        df = fetch_range_data(args.lat, args.lon, days=180)
        if os.path.exists(CSV_PATH):
            old = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
            combined = pd.concat([old, df]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
        else:
            combined = df
        combined.to_csv(CSV_PATH, index=False)
        create_feature_group(df)
        df_features = create_feature_group(combined)
        train_and_register_model(df_features)

        logger.info("Run-once complete.")
        return
    # ============================================================
#  STEP 4 — STREAMLIT DASHBOARD (with 3-day forecasting)
# ============================================================
# ============================================================
#  STEP 4 — STREAMLIT DASHBOARD
# ============================================================
def run_dashboard(model, df):
    if st is None:
        logger.error("Streamlit is not installed in this environment.")
        return

    st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")
    st.title("🌫️ Karachi AQI Prediction Dashboard")

    # ✅ Recompute derived features so they match model expectations
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["aqi_estimated"] = df.get("pm2_5", 0) * 0.5 + df.get("pm10", 0) * 0.3 + df.get("ozone", 0) * 0.2
    df["aqi_change_rate"] = df["aqi_estimated"].diff().fillna(0)

    latest = df.sort_values("datetime").tail(72)
    features = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "windspeed_10m", "surface_pressure", "cloud_cover",
                "pm2_5", "pm10", "ozone", "aqi_change_rate", "hour", "month"]
    X = latest[[c for c in features if c in latest.columns]]

    preds = model.predict(X) if not X.empty else []

    result_df = pd.DataFrame({
        "datetime": latest["datetime"],
        "Predicted AQI": preds
    }).set_index("datetime")

    st.line_chart(result_df)
    if len(preds):
        st.metric("🌤 Latest AQI Prediction", round(float(preds[-1]), 2))
    st.write("Data Source: Open-Meteo | Model: Random Forest | Feature Store: Hopsworks")



if __name__ == "__main__":
    main()
