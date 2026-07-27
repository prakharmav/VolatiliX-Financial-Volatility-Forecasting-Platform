# VolatiliX: Multi-Asset Financial Analytics & Volatility Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **Financial Analytics & Market Forecasting Platform** built to analyze, forecast, and backtest prices and volatility across Cryptocurrencies (`BTC-USD`) and Tech Equities (`NVDA`, `AAPL`). 

`VolatiliX` combines classical statistical econometrics (**ARIMA / SARIMA**), additive decomposition (**Facebook Prophet**), and deep learning sequence architectures (**Keras LSTM**) with custom technical indicators, mathematical volatility estimators, and a risk-adjusted algorithmic backtester.

---

## 📌 Key Features

- **Multi-Asset Ingestion & Caching**: Automated market data fetching via Yahoo Finance (`yfinance`) with local CSV caching for 24/7 Cryptocurrencies and Equities.
- **Advanced Technical Indicators**: Computes SMA (5, 10, 20, 50), EMA, RSI (14), Bollinger Bands (20-day, 2-std), and MACD signals.
- **Mathematical Volatility Estimators**: Implements Close-to-Close Volatility, **Parkinson Volatility** (High-Low spread), and **Garman-Klass Volatility** (full OHLC dynamics).
- **Multi-Model Machine Learning Suite**:
  - **ARIMA & SARIMA**: Autoregressive integrated moving average with seasonal cycles.
  - **Facebook Prophet**: Multi-seasonality and trend changepoint detection.
  - **Keras Deep LSTM**: Recurrent Neural Network with walk-forward sequence evaluation.
- **Quantitative Strategy Backtester**: Simulates signal-driven long/flat trading strategies, computing **Sharpe Ratio**, **Maximum Drawdown**, **Total Return**, and **Transaction Friction** vs. Buy-and-Hold benchmarks.
- **Interactive Web Dashboard**: Streamlit interface with Plotly candlestick overlays, asset preset selectors, and dynamic model comparison tables.

---

## 📂 Project Layout

```
.
├── config.yaml                # Default ticker (BTC-USD), 6-year date window, model hyperparams
├── requirements.txt           # Environment dependencies
├── src/
│   ├── config.py              # Configuration loader & directory manager
│   ├── data_loader.py         # Yahoo Finance downloader & local caching
│   ├── feature_engineering.py # SMA, EMA, RSI, Bollinger Bands, MACD, Returns
│   ├── volatility.py          # Close-to-Close, Parkinson, Garman-Klass estimators
│   ├── evaluation.py          # RMSE, MAE, MAPE, Direction Accuracy metrics
│   ├── backtest.py            # Long/flat signal-driven strategy backtester
│   ├── visualization.py       # Matplotlib & Plotly charting routines
│   ├── pipeline.py            # End-to-end train / evaluate / backtest pipeline
│   └── models/
│       ├── arima_model.py     # ARIMA & SARIMA statsmodels forecasters
│       ├── prophet_model.py   # Facebook Prophet additive model
│       └── lstm_model.py      # Keras Deep LSTM sequence neural network
├── scripts/
│   └── run_pipeline.py        # Command-line execution entry point
├── dashboard/
│   └── app.py                 # Streamlit web app with preset asset selector
├── notebooks/
│   └── exploratory_analysis.ipynb # Jupyter notebook for EDA & visualizations
├── tests/
│   └── test_basic.py          # Pytest suite with synthetic market data
├── screenshots/               # Application UI screenshots
└── results/                   # Generated metrics, plots, and saved models
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/prakharmav/VolatiliX-Financial-Volatility-Forecasting-Platform.git
cd VolatiliX-Financial-Volatility-Forecasting-Platform
```

### 2. Create & Activate Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`. Use the sidebar asset selector to toggle between **`BTC-USD`**, **`NVDA`**, **`AAPL`**, **`TSLA`**, **`MSFT`**, or enter any custom ticker symbol.

### 2. Run Pipeline via CLI
Run full training, evaluation, and backtesting pipelines from the command line:

```bash
# Run default configuration (BTC-USD, 2020 to 2026)
python scripts/run_pipeline.py

# Run NVIDIA without LSTM
python scripts/run_pipeline.py --ticker NVDA --no-lstm

# Run Apple with custom date range
python scripts/run_pipeline.py --ticker AAPL --start 2020-01-01 --end 2026-07-28
```

#### Pipeline Outputs Generated:
- `data/raw/{TICKER}.csv` — Raw market OHLCV data
- `data/processed/{TICKER}_features.csv` — Feature dataset with technical indicators
- `results/metrics.csv` — Model performance comparison table
- `results/backtest.csv` — Strategy equity curve vs. buy-and-hold
- `results/plots/*.png` — Indicator, forecast, and drawdown charts

---

## 📊 Dashboard Screenshots

### Interactive Streamlit Dashboard (`BTC-USD` Forecast & Metrics)
![BTC-USD Dashboard Forecast](screenshots/01-btc-forecast.png)

---

## 🧮 Quantitative Models & Formulas

### 1. Volatility Estimators (`src/volatility.py`)

- **Close-to-Close Volatility**:
  $$\sigma_{CC} = \sqrt{\frac{252}{N-1} \sum_{i=1}^{N} (r_i - \bar{r})^2}$$

- **Parkinson Volatility** (incorporates High-Low price ranges):
  $$\sigma_{P} = \sqrt{\frac{252}{4 \ln(2) N} \sum_{i=1}^{N} \left(\ln\frac{H_i}{L_i}\right)^2}$$

- **Garman-Klass Volatility** (incorporates OHLC prices):
  $$\sigma_{GK} = \sqrt{\frac{252}{N} \sum_{i=1}^{N} \left[ 0.5 \left(\ln\frac{H_i}{L_i}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_i}{O_i}\right)^2 \right]}$$

### 2. Forecasting Models Supported

| Model | Architecture / Library | Strengths | Use Case |
| :--- | :--- | :--- | :--- |
| **ARIMA** | `statsmodels` (Autoregressive Integrated Moving Average) | Fast, interpretable, linear trends | Short-term price forecasting |
| **SARIMA** | `statsmodels` (Seasonal ARIMA) | Captures periodic cyclicality | Assets with weekly seasonality |
| **Prophet** | `facebook/prophet` (Additive Regression) | Handles missing dates & holiday effects | Trend & multi-seasonality decomposition |
| **LSTM** | `TensorFlow / Keras` (Recurrent Deep Neural Net) | Learns complex non-linear sequence patterns | Long-range walk-forward sequence prediction |

### 3. Strategy Evaluation & Backtesting Metrics (`src/evaluation.py` & `src/backtest.py`)

- **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N} \sum (\hat{y}_i - y_i)^2}$
- **Mean Absolute Percentage Error (MAPE)**: $\frac{100\%}{N} \sum \left| \frac{y_i - \hat{y}_i}{y_i} \right|$
- **Direction Accuracy**: Percentage of correctly predicted up/down movement directions.
- **Sharpe Ratio**: Annualized risk-adjusted return metric: $SR = \frac{\bar{R}_p}{\sigma_p} \cdot \sqrt{252}$
- **Maximum Drawdown (MDD)**: Worst peak-to-trough equity decline percentage.

---

## 🧪 Unit Testing

Run the automated test suite to verify system integrity offline:

```bash
pytest tests/
```

Tests use synthetic market data generator functions to validate feature building, forecasting engines, metrics calculations, and backtesting pipelines.

---

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome!  
Distributed under the **MIT License**. See `LICENSE` for more information.

---

**Author**: [Prakhar Raj](https://github.com/prakharmav)  
*Multi-Asset Financial Analytics & Volatility Forecasting Platform*
