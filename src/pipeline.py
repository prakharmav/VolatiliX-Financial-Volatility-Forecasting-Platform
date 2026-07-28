from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import PROJECT_ROOT, ensure_dirs, load_config
from src.data_loader import get_data
from src.feature_engineering import build_features
from src.evaluation import evaluate_all
from src.backtest import (
    backtest_strategy,
    backtest_enhanced,
    performance_stats,
    performance_stats_enhanced,
    signal_from_forecast,
)
from src.strategy import build_enhanced_strategy
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

    # Default strategy config if not present (backward compat)
    strat_cfg = cfg.get("strategy", {})
    use_walk_forward = strat_cfg.get("use_walk_forward", True)
    refit_every = strat_cfg.get("refit_every", 20)

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

    # ----------------------------------------------------------------
    #  ARIMA — walk-forward or one-shot
    # ----------------------------------------------------------------
    print("[3/6] ARIMA...")
    arima = ARIMAForecaster(order=cfg["models"]["arima"]["order"]).fit(train_p)
    if use_walk_forward:
        arima_fc = arima.walk_forward_forecast(train_p, test_p, refit_every=refit_every)
        print(f"       (walk-forward, refit every {refit_every} steps)")
    else:
        arima_fc = arima.forecast(steps)
        arima_fc.index = test_p.index
    price_forecasts["ARIMA"] = arima_fc
    models_out["arima"] = arima

    # ----------------------------------------------------------------
    #  SARIMA — walk-forward or one-shot
    # ----------------------------------------------------------------
    print("[4/6] SARIMA...")
    sarima = SARIMAForecaster(
        order=cfg["models"]["sarima"]["order"],
        seasonal_order=cfg["models"]["sarima"]["seasonal_order"],
    ).fit(train_p)
    if use_walk_forward:
        sarima_fc = sarima.walk_forward_forecast(train_p, test_p, refit_every=refit_every)
        print(f"       (walk-forward, refit every {refit_every} steps)")
    else:
        sarima_fc = sarima.forecast(steps)
        sarima_fc.index = test_p.index
    price_forecasts["SARIMA"] = sarima_fc
    models_out["sarima"] = sarima

    # ----------------------------------------------------------------
    #  Prophet
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    #  LSTM
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    #  Evaluation
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    #  Original backtest (single best model, binary signal)
    # ----------------------------------------------------------------
    print("\n--- Original Backtest (single model, binary signal) ---")
    best_name = metrics["RMSE"].idxmin()
    best_fc = price_forecasts[best_name]
    sig = signal_from_forecast(best_fc, test_p)
    bt_original = backtest_strategy(
        test_p,
        sig,
        initial_cash=cfg["backtest"]["initial_cash"],
        transaction_cost=cfg["backtest"]["transaction_cost"],
    )
    stats_original = performance_stats(bt_original)
    print(f"Best model: {best_name}")
    for k, v in stats_original.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    bt_original.to_csv(Path(PROJECT_ROOT) / cfg["paths"]["results"] / "backtest.csv")
    plot_backtest_equity(bt_original, save_path=str(plots_dir / "backtest_equity.png"))

    # ----------------------------------------------------------------
    #  Enhanced backtest (ensemble + vol-sizing + stop-loss)
    # ----------------------------------------------------------------
    print("\n--- Enhanced Backtest (ensemble + filters + vol-sizing) ---")

    # Align features to test period
    test_features = feats.loc[test_p.index]

    position = build_enhanced_strategy(
        price_forecasts,
        test_p,
        test_features,
        min_agreement=strat_cfg.get("min_agreement", 0.5),
        vol_window=strat_cfg.get("vol_window", 21),
        target_vol=strat_cfg.get("target_vol", 0.15),
        max_position=strat_cfg.get("max_position", 1.0),
        rsi_upper=strat_cfg.get("rsi_upper", 70.0),
        stop_pct=strat_cfg.get("stop_pct", 0.05),
        cooldown=strat_cfg.get("cooldown", 3),
    )

    bt_enhanced = backtest_enhanced(
        test_p,
        position,
        initial_cash=cfg["backtest"]["initial_cash"],
        transaction_cost=cfg["backtest"]["transaction_cost"],
    )
    stats_enhanced = performance_stats_enhanced(bt_enhanced)
    print(f"Models in ensemble: {list(price_forecasts.keys())}")
    for k, v in stats_enhanced.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    bt_enhanced.to_csv(Path(PROJECT_ROOT) / cfg["paths"]["results"] / "backtest_enhanced.csv")

    # ----------------------------------------------------------------
    #  Comparison summary
    # ----------------------------------------------------------------
    print("\n=== COMPARISON ===")
    print(f"  Original  total_return: {stats_original['total_return']*100:.2f}%  "
          f"sharpe: {stats_original['sharpe']:.2f}  max_dd: {stats_original['max_drawdown']*100:.2f}%")
    print(f"  Enhanced  total_return: {stats_enhanced['total_return']*100:.2f}%  "
          f"sharpe: {stats_enhanced['sharpe']:.2f}  max_dd: {stats_enhanced['max_drawdown']*100:.2f}%  "
          f"win_rate: {stats_enhanced['win_rate']*100:.1f}%  "
          f"profit_factor: {stats_enhanced['profit_factor']:.2f}")

    # ----------------------------------------------------------------
    #  Save models
    # ----------------------------------------------------------------
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
        "backtest": bt_original,
        "backtest_stats": stats_original,
        "backtest_enhanced": bt_enhanced,
        "backtest_enhanced_stats": stats_enhanced,
        "best_model": best_name,
        "features": feats,
    }


if __name__ == "__main__":
    run_pipeline()
