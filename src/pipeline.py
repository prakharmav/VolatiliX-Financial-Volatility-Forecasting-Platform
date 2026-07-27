from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import PROJECT_ROOT, ensure_dirs, load_config
from src.data_loader import get_data
from src.feature_engineering import build_features
from src.evaluation import evaluate_all
from src.backtest import backtest_strategy, performance_stats, signal_from_forecast
from src.models.arima_model import ARIMAForecaster, SARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.visualization import (
    plot_backtest_equity,
    plot_forecast_vs_actual,
    plot_price_with_indicators,
)


def train_test_split(series: pd.Series, train_ratio: float) -> tuple[pd.Series, pd.Series]:
    n = int(len(series) * train_ratio)
    return series.iloc[:n], series.iloc[n:]


def run_pipeline(cfg: dict | None = None, include_lstm: bool = True, include_prophet: bool = True) -> dict:
    cfg = cfg or load_config()
    ensure_dirs(cfg)

    print(f"[1/6] Loading data for {cfg['data']['ticker']}...")
    raw = get_data(cfg)

    print("[2/6] Building features...")
    feats = build_features(raw, cfg)
    feats.to_csv(Path(PROJECT_ROOT) / cfg["data"]["processed_path"] / f"{cfg['data']['ticker']}_features.csv")

    plots_dir = Path(PROJECT_ROOT) / cfg["paths"]["plots"]
    plot_price_with_indicators(feats, save_path=str(plots_dir / "price_indicators.png"))

    target_price = feats["Close"]
    train_p, test_p = train_test_split(target_price, cfg["split"]["train_ratio"])
    steps = len(test_p)
    print(f"     train={len(train_p)}  test={len(test_p)}")

    price_forecasts: dict[str, pd.Series] = {}
    models_out: dict = {}

    print("[3/6] ARIMA...")
    arima = ARIMAForecaster(order=cfg["models"]["arima"]["order"]).fit(train_p)
    arima_fc = arima.forecast(steps)
    arima_fc.index = test_p.index
    price_forecasts["ARIMA"] = arima_fc
    models_out["arima"] = arima

    print("[4/6] SARIMA...")
    sarima = SARIMAForecaster(
        order=cfg["models"]["sarima"]["order"],
        seasonal_order=cfg["models"]["sarima"]["seasonal_order"],
    ).fit(train_p)
    sarima_fc = sarima.forecast(steps)
    sarima_fc.index = test_p.index
    price_forecasts["SARIMA"] = sarima_fc
    models_out["sarima"] = sarima

    if include_prophet:
        print("[5/6] Prophet...")
        try:
            prop = ProphetForecaster(**cfg["models"]["prophet"]).fit(train_p)
            prop_fc = prop.forecast(steps)
            prop_fc = pd.Series(prop_fc.values[: len(test_p)], index=test_p.index, name="forecast")
            price_forecasts["Prophet"] = prop_fc
            models_out["prophet"] = prop
        except Exception as exc:
            print(f"     Prophet skipped: {exc}")

    if include_lstm:
        print("[6/6] LSTM...")
        try:
            from src.models.lstm_model import LSTMForecaster

            lstm = LSTMForecaster(**cfg["models"]["lstm"]).fit(train_p, verbose=0)
            lstm_fc = lstm.predict_on_test(train_p, test_p)
            price_forecasts["LSTM"] = lstm_fc
            models_out["lstm"] = lstm
        except Exception as exc:
            print(f"     LSTM skipped: {exc}")

    print("\nEvaluating forecasts...")
    metrics = evaluate_all(price_forecasts, test_p)
    print(metrics.round(4))
    metrics.to_csv(Path(PROJECT_ROOT) / cfg["paths"]["results"] / "metrics.csv")

    plot_forecast_vs_actual(
        train_p,
        test_p,
        price_forecasts,
        title=f"{cfg['data']['ticker']} — Close Price Forecasts",
        save_path=str(plots_dir / "forecasts.png"),
    )

    print("\nBacktesting top model...")
    best_name = metrics["RMSE"].idxmin()
    best_fc = price_forecasts[best_name]
    sig = signal_from_forecast(best_fc, test_p)
    bt = backtest_strategy(
        test_p,
        sig,
        initial_cash=cfg["backtest"]["initial_cash"],
        transaction_cost=cfg["backtest"]["transaction_cost"],
    )
    stats = performance_stats(bt)
    print(f"Best model: {best_name}")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    bt.to_csv(Path(PROJECT_ROOT) / cfg["paths"]["results"] / "backtest.csv")
    plot_backtest_equity(bt, save_path=str(plots_dir / "backtest_equity.png"))

    out_models = Path(PROJECT_ROOT) / cfg["paths"]["models"]
    for name, model in models_out.items():
        if name == "lstm":
            try:
                model.model_.save(out_models / "lstm.keras")
            except Exception:
                pass
        else:
            try:
                joblib.dump(model, out_models / f"{name}.joblib")
            except Exception:
                pass

    return {
        "metrics": metrics,
        "forecasts": price_forecasts,
        "backtest": bt,
        "backtest_stats": stats,
        "best_model": best_name,
        "features": feats,
    }


if __name__ == "__main__":
    run_pipeline()
