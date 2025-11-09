# 🌫️ AQI Prediction System

**Automated End-to-End Data Pipeline with GitHub Actions and Hopsworks**

An intelligent air quality intelligence pipeline that fetches real-time weather and pollutant data, generates features, trains ML models, and visualizes predictions on a live dashboard.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/syahra712/AQI10Pearls/actions)

---

## 🎯 Quick Start

```bash
# Clone the repository
git clone https://github.com/syahra712/AQI10Pearls.git
cd AQI10Pearls

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export HOPSWORKS_API_KEY=your_api_key
export HOPSWORKS_PROJECT=your_project_name
export HOPSWORKS_HOST=c.app.hopsworks.ai

# Run the pipeline
python hithere.py --mode fetch --hours 24  # Fetch latest data
python hithere.py --mode train              # Train model
python hithere.py --mode dashboard          # View dashboard
```

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Data Pipeline](#-data-pipeline)
- [Automation](#-automation--cicd)
- [Setup & Configuration](#-setup--configuration)
- [Usage](#-usage)
- [Technologies](#-technologies)
- [Scalability](#-scalability--future-improvements)
- [Contributing](#-contributing)

---

## ✨ Features

- **Real-time Data Ingestion** — Hourly weather and air quality data from Open-Meteo APIs
- **Smart Feature Engineering** — Composite AQI metrics and temporal features
- **ML-Powered Predictions** — Random Forest model with ~0.87 R² score
- **Centralized Feature Store** — Hopsworks integration for versioned feature management
- **Automated Retraining** — Daily model updates via GitHub Actions
- **Live Dashboard** — Streamlit visualization of predictions and metrics
- **Production-Ready** — Secure credential management and error handling

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
│  Open-Meteo Weather API  │  Open-Meteo Air Quality API          │
└────────────────┬─────────────────────────────────┬──────────────┘
                 │                                 │
                 └─────────────┬───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Data Ingestion    │
                    │   (hithere.py)      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────┐
                    │ Feature Engineering    │
                    │ • AQI Estimation       │
                    │ • Temporal Features    │
                    │ • Rate of Change       │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │  Hopsworks Feature     │
                    │  Store (Versioned)     │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   Model Training       │
                    │ RandomForestRegressor  │
                    │   (RMSE: 5.83)         │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │  Hopsworks Model       │
                    │  Registry (Versioned)  │
                    └──────────┬─────────────┘
                               │
                ┌──────────────┴───────────────┐
                │                              │
     ┌──────────▼──────────┐       ┌──────────▼──────────┐
     │   Streamlit         │       │   GitHub Actions    │
     │   Dashboard         │       │   Automation        │
     └─────────────────────┘       └─────────────────────┘
```

---

## 🧠 Data Pipeline

### Step 1: Data Fetching
- Fetches **6 months of historical data** on initialization
- Continues with **hourly real-time updates**
- 7-day rolling windows to avoid API rate limits
- Covers: Karachi (24.8607°N, 67.0011°E)

**Data Points:**
- Temperature, Humidity, Dew Point, Wind Speed
- PM2.5, PM10, Ozone (O₃), Nitrogen Dioxide (NO₂), Carbon Monoxide (CO)

### Step 2: Feature Engineering
| Feature | Description |
|---------|-------------|
| `aqi_estimated` | Composite: `0.5*PM2.5 + 0.3*PM10 + 0.2*O₃` |
| `aqi_change_rate` | Hourly AQI rate of change |
| `hour`, `day`, `month` | Temporal features from timestamp |
| Weather features | Temperature, humidity, wind speed, cloud cover |

### Step 3: Model Training
- **Algorithm:** Random Forest Regressor
- **Training Split:** 80/20 (train/test)
- **Performance:**
  - RMSE: ~5.83
  - R² Score: ~0.87
- **Auto-versioning** in Hopsworks Model Registry

### Step 4: Visualization
- Real-time line chart of AQI predictions
- Live metric tiles with latest predictions
- Integrated Hopsworks feature retrieval

---

## 🤖 Automation & CI/CD

GitHub Actions automatically manages the entire pipeline:

```yaml
┌─────────────────────────────────────────────────────┐
│         GitHub Actions Workflows                    │
├─────────────────────────────────────────────────────┤
│ Job              │ Schedule      │ Action            │
├──────────────────┼───────────────┼──────────────────┤
│ hourly-fetch     │ Every hour    │ Fetch new data   │
│ daily-train      │ Daily 00:30   │ Retrain model    │
│ dashboard-review │ Manual trigger│ Preview dashboard│
└─────────────────────────────────────────────────────┘
```

### Workflow File
`.github/workflows/aqi_automation.yml`

**Executed Commands:**
```bash
python3 hithere.py --mode fetch --hours 24
python3 hithere.py --mode train
```

---

## 🔐 Setup & Configuration

### Prerequisites
- Python 3.12+
- GitHub Account with Actions enabled
- Hopsworks Project ([Sign up free](https://www.hopsworks.ai/))

### Environment Variables

Create a `.env` file (or set as GitHub Secrets):

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=your_project_name
HOPSWORKS_HOST=c.app.hopsworks.ai
```

### GitHub Secrets Setup

1. Go to **Settings → Secrets and variables → Actions**
2. Add the following secrets:
   - `HOPSWORKS_API_KEY`
   - `HOPSWORKS_PROJECT`
   - `HOPSWORKS_HOST`

---

## 🚀 Usage

### Fetch Latest Data
```bash
python hithere.py --mode fetch --hours 24
```
Fetches the last 24 hours of weather and AQ data, pushes to Hopsworks.

### Train Model
```bash
python hithere.py --mode train
```
Trains RandomForest on latest features, registers new model version in Hopsworks.

### View Dashboard
```bash
streamlit run hithere.py -- --mode dashboard
```
Opens interactive Streamlit dashboard at `http://localhost:8501`

### Combined Run
```bash
python hithere.py --mode fetch --hours 24 && \
python hithere.py --mode train && \
streamlit run hithere.py -- --mode dashboard
```

---

## 📦 Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.12 | Core implementation |
| **Data Fetching** | Requests, Open-Meteo API | Real-time weather & AQ data |
| **Data Processing** | Pandas, NumPy | Preprocessing & feature engineering |
| **ML Model** | scikit-learn | RandomForestRegressor for predictions |
| **Feature Store** | Hopsworks 4.2 | Centralized feature management & versioning |
| **Model Registry** | Hopsworks | Model tracking & versioning |
| **Dashboard** | Streamlit | Interactive visualization |
| **CI/CD** | GitHub Actions | Automated pipeline orchestration |
| **Version Control** | Git | Code & workflow management |

---

## 📊 Model Performance

```
Model: Random Forest Regressor
Target: AQI Estimation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RMSE:       5.83
R² Score:   0.87
MAE:        4.12
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: Production Ready ✓
```

---

## 🔹 Scalability & Future Improvements

### Planned Enhancements
- [ ] **Ensemble Models** — XGBoost & LightGBM for improved accuracy
- [ ] **SHAP Explainability** — Interpretability metrics for predictions
- [ ] **Automated Reports** — Daily/weekly AQI trend HTML/CSV reports
- [ ] **Multi-City Coverage** — Lahore, Islamabad, Peshawar support
- [ ] **Dockerization** — Container deployment on AWS Lambda / GCP Cloud Run
- [ ] **Alert System** — Twilio/Email notifications for hazardous AQI levels
- [ ] **Real-time Forecasting** — 24-48 hour AQI forecasts

---

## 📂 Project Structure

```
AQI10Pearls/
├── hithere.py                          # Core pipeline script
├── requirements.txt                    # Python dependencies
├── model/
│   └── rf_model.pkl                   # Trained model artifact
├── karachi_weather_pollutants.csv     # Feature dataset (auto-updating)
├── .github/
│   └── workflows/
│       └── aqi_automation.yml         # GitHub Actions workflow
├── .gitignore
├── LICENSE
└── README.md                          # This file
```

---

## 🌍 Sustainability & SDG Impact

This project aligns with **UN Sustainable Development Goal 13 (Climate Action)** by:

✅ Promoting data-driven air quality monitoring  
✅ Raising citizen awareness of pollution patterns  
✅ Enabling proactive climate measures by government agencies  
✅ Supporting environmental research and policy-making

---

## 📚 Resources

- **Open-Meteo API Docs:** [https://open-meteo.com/en/docs](https://open-meteo.com/en/docs)
- **Hopsworks Documentation:** [https://docs.hopsworks.ai/](https://docs.hopsworks.ai/)
- **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **GitHub Actions Docs:** [https://docs.github.com/en/actions](https://docs.github.com/en/actions)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Support & Questions

- 📧 Open an [Issue](https://github.com/syahra712/AQI10Pearls/issues) for bug reports
- 💬 Start a [Discussion](https://github.com/syahra712/AQI10Pearls/discussions) for questions
- ⭐ Star the repo if you find it useful!

---

## 🔗 Links

| Resource | Link |
|----------|------|
| Hopsworks Project | [https://c.app.hopsworks.ai/p/1286309](https://c.app.hopsworks.ai/p/1286309) |
| GitHub Repository | [https://github.com/syahra712/AQI10Pearls](https://github.com/syahra712/AQI10Pearls) |

---

**Made with ❤️ for cleaner air and better climate action**
