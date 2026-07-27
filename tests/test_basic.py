from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


def _synthetic_ohlcv(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": vol},
        index=idx,
    )


def test_features():
    from src.feature_engineering import build_features

    df = _synthetic_ohlcv()
    cfg = {
        "features": {
            "ma_windows": [5, 20],
            "rsi_period": 14,
            "bollinger_window": 20,
            "bollinger_std": 2,
            "volatility_window": 21,
        }
    }
    feats = build_features(df, cfg)
    for col in ("sma_5", "sma_20", "rsi", "bb_upper", "bb_lower", "macd", "volatility"):
        assert col in feats.columns
    assert len(feats) > 0
    assert feats["rsi"].between(0, 100).all()


def test_arima_forecasts():
    from src.models.arima_model import ARIMAForecaster

    df = _synthetic_ohlcv(n=300)
    m = ARIMAForecaster(order=(1, 1, 1)).fit(df["Close"])
    fc = m.forecast(steps=10)
    assert len(fc) == 10
    assert fc.notna().all()


def test_evaluation_metrics():
    from src.evaluation import evaluate, rmse, mae

    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    p = pd.Series([1.1, 1.9, 3.2, 3.8])
    assert rmse(y, p) > 0
    assert mae(y, p) > 0
    out = evaluate(y, p, "test")
    assert out["model"] == "test"


def test_backtest_runs():
    from src.backtest import backtest_strategy, performance_stats, signal_from_forecast

    df = _synthetic_ohlcv(n=200)
    prices = df["Close"]
    fc = prices * (1 + np.random.default_rng(0).normal(0, 0.01, len(prices)))
    sig = signal_from_forecast(fc, prices)
    bt = backtest_strategy(prices, sig)
    stats = performance_stats(bt)
    assert "total_return" in stats
    assert len(bt) == len(prices)
