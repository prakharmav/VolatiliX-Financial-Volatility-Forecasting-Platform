from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtest import backtest_strategy, performance_stats, signal_from_forecast
from src.config import load_config
from src.data_loader import download_data
from src.evaluation import evaluate_all
from src.feature_engineering import build_features
from src.models.arima_model import ARIMAForecaster, SARIMAForecaster


st.set_page_config(page_title="Stock Volatility Forecasting", layout="wide", page_icon="📈")
st.title("Stock Price Volatility Forecasting")
st.caption("ARIMA / SARIMA / Prophet / LSTM with technical indicators and backtesting")


@st.cache_data(show_spinner=False)
def _load(ticker: str, start: str, end: str) -> pd.DataFrame:
    return download_data(ticker, start, end, save_dir=None)


@st.cache_data(show_spinner=False)
def _features(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return build_features(raw, cfg)


cfg = load_config()

with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Asset Selection")
    preset_choice = st.selectbox(
        "Select Asset / Ticker",
        ["BTC-USD (Bitcoin)", "NVDA (NVIDIA)", "AAPL (Apple)", "TSLA (Tesla)", "MSFT (Microsoft)", "Custom Ticker"],
        index=0
    )
    
    if preset_choice == "Custom Ticker":
        ticker = st.text_input("Custom Ticker Symbol", value=cfg["data"]["ticker"]).upper().strip()
    else:
        ticker = preset_choice.split(" ")[0]

    start = st.date_input("Start date", value=pd.to_datetime(cfg["data"]["start_date"]))
    end = st.date_input("End date", value=pd.to_datetime(cfg["data"]["end_date"]))
    train_ratio = st.slider("Train ratio", 0.5, 0.95, cfg["split"]["train_ratio"], 0.05)

    st.subheader("Models")
    use_arima = st.checkbox("ARIMA", True)
    use_sarima = st.checkbox("SARIMA", True)
    use_prophet = st.checkbox("Prophet", False)
    use_lstm = st.checkbox("LSTM (slower)", False)

    st.subheader("Backtest")
    cost = st.number_input("Transaction cost", value=cfg["backtest"]["transaction_cost"], step=0.0005, format="%.4f")
    cash = st.number_input("Initial cash", value=float(cfg["backtest"]["initial_cash"]), step=1000.0)

    run_btn = st.button("Run forecast", type="primary", use_container_width=True)


if not run_btn:
    st.info("Configure parameters in the sidebar and click **Run forecast**.")
    st.stop()

with st.spinner(f"Downloading {ticker}..."):
    try:
        raw = _load(ticker, str(start), str(end))
    except Exception as exc:
        st.error(f"Failed to download data: {exc}")
        st.stop()

feats = _features(raw, cfg)
st.success(f"Loaded {len(feats)} rows for {ticker}")

tab_overview, tab_indicators, tab_forecast, tab_backtest = st.tabs(
    ["Overview", "Indicators", "Forecasts", "Backtest"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    last = feats.iloc[-1]
    prev = feats.iloc[-2]
    c1.metric("Last close", f"${last['Close']:.2f}", f"{(last['Close']/prev['Close']-1)*100:.2f}%")
    c2.metric("Volatility (ann.)", f"{last['volatility']*100:.2f}%")
    c3.metric("RSI(14)", f"{last['rsi']:.1f}")
    c4.metric("Volume", f"{int(last['Volume']):,}")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03
    )
    fig.add_trace(
        go.Candlestick(
            x=feats.index,
            open=feats["Open"],
            high=feats["High"],
            low=feats["Low"],
            close=feats["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    if "sma_20" in feats:
        fig.add_trace(go.Scatter(x=feats.index, y=feats["sma_20"], name="SMA 20", line=dict(width=1)), row=1, col=1)
    if "sma_50" in feats:
        fig.add_trace(go.Scatter(x=feats.index, y=feats["sma_50"], name="SMA 50", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=feats.index, y=feats["Volume"], name="Volume", marker_color="lightgray"), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tab_indicators:
    col1, col2 = st.columns(2)
    with col1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=feats.index, y=feats["Close"], name="Close"))
        f.add_trace(go.Scatter(x=feats.index, y=feats["bb_upper"], name="BB Upper", line=dict(dash="dash")))
        f.add_trace(go.Scatter(x=feats.index, y=feats["bb_lower"], name="BB Lower", line=dict(dash="dash"), fill="tonexty", fillcolor="rgba(100,100,200,0.1)"))
        f.update_layout(title="Bollinger Bands", height=400)
        st.plotly_chart(f, use_container_width=True)

        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=feats.index, y=feats["volatility"] * 100, name="Volatility (ann %)", line=dict(color="orange")))
        f3.update_layout(title="Realised Volatility (annualised %)", height=400)
        st.plotly_chart(f3, use_container_width=True)

    with col2:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=feats.index, y=feats["rsi"], name="RSI", line=dict(color="purple")))
        f2.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        f2.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
        f2.update_layout(title="RSI (14)", yaxis_range=[0, 100], height=400)
        st.plotly_chart(f2, use_container_width=True)

        f4 = go.Figure()
        f4.add_trace(go.Scatter(x=feats.index, y=feats["macd"], name="MACD"))
        f4.add_trace(go.Scatter(x=feats.index, y=feats["macd_signal"], name="Signal"))
        f4.add_trace(go.Bar(x=feats.index, y=feats["macd_hist"], name="Hist", marker_color="lightblue"))
        f4.update_layout(title="MACD", height=400)
        st.plotly_chart(f4, use_container_width=True)


price = feats["Close"]
n_train = int(len(price) * train_ratio)
train, test = price.iloc[:n_train], price.iloc[n_train:]
steps = len(test)

forecasts: dict[str, pd.Series] = {}

with tab_forecast:
    progress = st.progress(0.0)
    n_models = sum([use_arima, use_sarima, use_prophet, use_lstm]) or 1
    done = 0

    if use_arima:
        with st.spinner("Fitting ARIMA..."):
            try:
                m = ARIMAForecaster(order=cfg["models"]["arima"]["order"]).fit(train)
                fc = m.forecast(steps)
                fc.index = test.index
                forecasts["ARIMA"] = fc
            except Exception as exc:
                st.warning(f"ARIMA failed: {exc}")
        done += 1
        progress.progress(done / n_models)

    if use_sarima:
        with st.spinner("Fitting SARIMA..."):
            try:
                m = SARIMAForecaster(
                    order=cfg["models"]["sarima"]["order"],
                    seasonal_order=cfg["models"]["sarima"]["seasonal_order"],
                ).fit(train)
                fc = m.forecast(steps)
                fc.index = test.index
                forecasts["SARIMA"] = fc
            except Exception as exc:
                st.warning(f"SARIMA failed: {exc}")
        done += 1
        progress.progress(done / n_models)

    if use_prophet:
        with st.spinner("Fitting Prophet..."):
            try:
                from src.models.prophet_model import ProphetForecaster

                m = ProphetForecaster(**cfg["models"]["prophet"]).fit(train)
                fc = m.forecast(steps)
                fc = pd.Series(fc.values[:steps], index=test.index, name="Prophet")
                forecasts["Prophet"] = fc
            except Exception as exc:
                st.warning(f"Prophet failed: {exc}")
        done += 1
        progress.progress(done / n_models)

    if use_lstm:
        with st.spinner("Training LSTM (this can take a minute)..."):
            try:
                from src.models.lstm_model import LSTMForecaster

                m = LSTMForecaster(**cfg["models"]["lstm"]).fit(train, verbose=0)
                fc = m.predict_on_test(train, test)
                forecasts["LSTM"] = fc
            except Exception as exc:
                st.warning(f"LSTM failed: {exc}")
        done += 1
        progress.progress(done / n_models)

    progress.empty()

    if not forecasts:
        st.error("No models produced forecasts.")
        st.stop()

    f = go.Figure()
    f.add_trace(go.Scatter(x=train.index[-200:], y=train.values[-200:], name="Train (recent)", line=dict(color="lightgrey")))
    f.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual", line=dict(color="black", width=2)))
    for name, pred in forecasts.items():
        f.add_trace(go.Scatter(x=pred.index, y=pred.values, name=name, line=dict(width=1.5)))
    f.update_layout(title=f"{ticker} — Close Price Forecasts", height=550)
    st.plotly_chart(f, use_container_width=True)

    metrics = evaluate_all(forecasts, test)
    st.subheader("Evaluation Metrics")
    st.dataframe(metrics.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r", subset=["RMSE", "MAE", "MAPE"]))


with tab_backtest:
    best = metrics["RMSE"].idxmin()
    chosen = st.selectbox("Choose model for backtest", list(forecasts.keys()), index=list(forecasts.keys()).index(best))

    fc = forecasts[chosen]
    sig = signal_from_forecast(fc, test)
    bt = backtest_strategy(test, sig, initial_cash=cash, transaction_cost=cost)
    stats = performance_stats(bt)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy return", f"{stats['total_return']*100:.2f}%")
    c2.metric("Buy & hold", f"{stats['buy_hold_return']*100:.2f}%")
    c3.metric("Sharpe", f"{stats['sharpe']:.2f}")
    c4.metric("Max drawdown", f"{stats['max_drawdown']*100:.2f}%")

    f = go.Figure()
    f.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name=f"{chosen} strategy", line=dict(width=2)))
    f.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="Buy & hold", line=dict(dash="dash")))
    f.update_layout(title="Equity Curve", height=500, yaxis_title="Portfolio value")
    st.plotly_chart(f, use_container_width=True)

    st.caption(f"Trades: {stats['num_trades']}  •  Transaction cost: {cost:.4f}")
