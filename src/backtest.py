from __future__ import annotations

import numpy as np
import pandas as pd


def signal_from_forecast(forecast: pd.Series, prices: pd.Series) -> pd.Series:
    aligned = forecast.reindex(prices.index).ffill()
    return (aligned.shift(-1) > prices).astype(int)


def backtest_strategy(
    prices: pd.Series,
    signal: pd.Series,
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    df = pd.DataFrame({"price": prices, "signal": signal.reindex(prices.index).fillna(0)})
    df["position"] = df["signal"].shift(1).fillna(0)
    df["daily_return"] = df["price"].pct_change().fillna(0)

    trades = df["position"].diff().abs().fillna(0)
    costs = trades * transaction_cost
    df["strategy_return"] = df["position"] * df["daily_return"] - costs

    df["equity"] = (1 + df["strategy_return"]).cumprod() * initial_cash
    df["buy_hold_equity"] = (1 + df["daily_return"]).cumprod() * initial_cash
    return df


def performance_stats(bt: pd.DataFrame) -> dict:
    rets = bt["strategy_return"].dropna()
    bh_rets = bt["daily_return"].dropna()
    if len(rets) == 0:
        return {}

    cum = bt["equity"].iloc[-1] / bt["equity"].iloc[0] - 1
    bh_cum = bt["buy_hold_equity"].iloc[-1] / bt["buy_hold_equity"].iloc[0] - 1
    ann_factor = np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * ann_factor) if rets.std() > 0 else 0.0
    bh_sharpe = (bh_rets.mean() / bh_rets.std() * ann_factor) if bh_rets.std() > 0 else 0.0

    eq = bt["equity"]
    drawdown = (eq / eq.cummax() - 1).min()

    return {
        "total_return": float(cum),
        "buy_hold_return": float(bh_cum),
        "sharpe": float(sharpe),
        "buy_hold_sharpe": float(bh_sharpe),
        "max_drawdown": float(drawdown),
        "num_trades": int(bt["position"].diff().abs().fillna(0).sum()),
    }


# ---------------------------------------------------------------------------
#  Enhanced backtest — supports fractional positions from vol-sizing
# ---------------------------------------------------------------------------

def backtest_enhanced(
    prices: pd.Series,
    position_size: pd.Series,
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """Backtest with fractional position sizing (0 to 1.0).

    Unlike :func:`backtest_strategy` which takes binary signals,
    this accepts a continuous position-size series produced by the
    enhanced strategy engine.

    Parameters
    ----------
    prices : pd.Series
        Close prices for the test period.
    position_size : pd.Series
        Fractional position (0 = flat, 1 = fully invested).
    initial_cash : float
        Starting capital.
    transaction_cost : float
        Round-trip cost per unit traded.

    Returns
    -------
    pd.DataFrame
        Columns: price, position, daily_return, strategy_return,
        equity, buy_hold_equity.
    """
    df = pd.DataFrame({
        "price": prices,
        "position": position_size.reindex(prices.index).fillna(0).shift(1).fillna(0),
    })
    df["daily_return"] = df["price"].pct_change().fillna(0)

    # Transaction costs proportional to the change in position size
    trades = df["position"].diff().abs().fillna(0)
    costs = trades * transaction_cost
    df["strategy_return"] = df["position"] * df["daily_return"] - costs

    df["equity"] = (1 + df["strategy_return"]).cumprod() * initial_cash
    df["buy_hold_equity"] = (1 + df["daily_return"]).cumprod() * initial_cash
    return df


def performance_stats_enhanced(bt: pd.DataFrame) -> dict:
    """Extended performance statistics including win rate, profit factor, etc."""
    rets = bt["strategy_return"].dropna()
    bh_rets = bt["daily_return"].dropna()
    if len(rets) == 0:
        return {}

    cum = bt["equity"].iloc[-1] / bt["equity"].iloc[0] - 1
    bh_cum = bt["buy_hold_equity"].iloc[-1] / bt["buy_hold_equity"].iloc[0] - 1
    n_days = len(rets)
    ann_factor = np.sqrt(252)

    # Sharpe ratio
    sharpe = (rets.mean() / rets.std() * ann_factor) if rets.std() > 0 else 0.0
    bh_sharpe = (bh_rets.mean() / bh_rets.std() * ann_factor) if bh_rets.std() > 0 else 0.0

    # Annualized return
    ann_return = (1 + cum) ** (252 / max(n_days, 1)) - 1

    # Max drawdown
    eq = bt["equity"]
    running_max = eq.cummax()
    drawdown_series = eq / running_max - 1
    max_dd = drawdown_series.min()

    # Calmar ratio (annualized return / |max drawdown|)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate — fraction of days with positive strategy returns (when in position)
    in_trade = bt["position"] > 0
    trade_rets = rets[in_trade]
    win_rate = (trade_rets > 0).mean() if len(trade_rets) > 0 else 0.0

    # Profit factor — gross profits / gross losses
    gross_profit = trade_rets[trade_rets > 0].sum()
    gross_loss = abs(trade_rets[trade_rets < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Number of trades (changes in position)
    num_trades = int(bt["position"].diff().abs().fillna(0).gt(0.01).sum())

    return {
        "total_return": float(cum),
        "buy_hold_return": float(bh_cum),
        "annualized_return": float(ann_return),
        "sharpe": float(sharpe),
        "buy_hold_sharpe": float(bh_sharpe),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "num_trades": num_trades,
    }
