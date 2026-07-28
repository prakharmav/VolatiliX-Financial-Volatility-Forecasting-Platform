"""Enhanced trading strategy with ensemble voting, volatility sizing, and risk filters."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  Ensemble signal — majority voting across multiple model forecasts
# ---------------------------------------------------------------------------

def ensemble_signal(
    forecasts: dict[str, pd.Series],
    prices: pd.Series,
    min_agreement: float = 0.5,
) -> pd.Series:
    """Generate a consensus signal from multiple model forecasts.

    For each day the individual model signals are computed
    (forecast_tomorrow > price_today → 1, else 0) and combined via majority
    voting.  A buy signal is emitted only when at least *min_agreement*
    fraction of models agree.

    Parameters
    ----------
    forecasts : dict[str, pd.Series]
        Model name → forecast series (indexed like *prices*).
    prices : pd.Series
        Actual close prices.
    min_agreement : float
        Fraction of models that must agree for a buy signal (0-1).
        Default 0.5 = simple majority.

    Returns
    -------
    pd.Series
        Binary signal (1 = buy, 0 = flat/sell), indexed like *prices*.
    """
    if not forecasts:
        return pd.Series(0, index=prices.index, name="signal")

    signals = pd.DataFrame(index=prices.index)
    for name, fc in forecasts.items():
        aligned = fc.reindex(prices.index).ffill()
        # signal: if model thinks tomorrow is higher than today → buy
        signals[name] = (aligned.shift(-1) > prices).astype(int)

    vote_frac = signals.mean(axis=1)
    consensus = (vote_frac >= min_agreement).astype(int)
    consensus.name = "signal"
    return consensus


# ---------------------------------------------------------------------------
# 2.  Volatility-scaled position sizing (risk-parity style)
# ---------------------------------------------------------------------------

def volatility_position_sizing(
    signal: pd.Series,
    prices: pd.Series,
    vol_window: int = 21,
    target_vol: float = 0.15,
    max_position: float = 1.0,
) -> pd.Series:
    """Scale positions inversely to recent realised volatility.

    When volatility is high, the position is scaled down so that the
    portfolio's expected daily risk stays near *target_vol / sqrt(252)*.

    Parameters
    ----------
    signal : pd.Series
        Binary buy/flat signal.
    prices : pd.Series
        Close prices.
    vol_window : int
        Rolling window for realised vol estimation.
    target_vol : float
        Target annualised volatility (0.15 = 15%).
    max_position : float
        Cap on the maximum position size (1.0 = fully invested).

    Returns
    -------
    pd.Series
        Fractional position sizes (0 to *max_position*), indexed like *prices*.
    """
    log_ret = np.log(prices / prices.shift(1))
    realised_vol = log_ret.rolling(window=vol_window, min_periods=5).std() * np.sqrt(252)
    realised_vol = realised_vol.replace(0, np.nan).ffill().bfill()

    # Scale: position = target_vol / realised_vol, clamped to [0, max_position]
    raw_size = (target_vol / realised_vol).clip(0, max_position)
    sized = signal * raw_size
    sized.name = "position_size"
    return sized


# ---------------------------------------------------------------------------
# 3.  RSI + Bollinger Band confirmation filter
# ---------------------------------------------------------------------------

def rsi_bollinger_filter(
    signal: pd.Series,
    features: pd.DataFrame,
    rsi_upper: float = 70.0,
    rsi_lower: float = 30.0,
) -> pd.Series:
    """Suppress buy signals when RSI is overbought or price is above upper BB.

    Parameters
    ----------
    signal : pd.Series
        Raw buy/flat signal.
    features : pd.DataFrame
        Must contain columns 'rsi', 'Close', and 'bb_upper'.
    rsi_upper : float
        RSI above this value → suppress buy signal.
    rsi_lower : float
        (Reserved for future use — short-side filter.)

    Returns
    -------
    pd.Series
        Filtered signal.
    """
    filtered = signal.copy()
    rsi = features["rsi"].reindex(signal.index).ffill()
    bb_upper = features["bb_upper"].reindex(signal.index).ffill()
    close = features["Close"].reindex(signal.index).ffill()

    # Suppress buy when RSI is overbought
    overbought = rsi > rsi_upper
    filtered[overbought] = 0

    # Suppress buy when price is above upper Bollinger Band
    above_bb = close > bb_upper
    filtered[above_bb] = 0

    filtered.name = "signal"
    return filtered


# ---------------------------------------------------------------------------
# 4.  Trailing stop-loss
# ---------------------------------------------------------------------------

def apply_trailing_stop(
    prices: pd.Series,
    signal: pd.Series,
    stop_pct: float = 0.05,
    cooldown: int = 3,
) -> pd.Series:
    """Apply a trailing stop-loss and post-exit cooldown.

    While in a position, track the peak price.  If the price drops
    *stop_pct* from the peak, exit and remain flat for *cooldown* bars.

    Parameters
    ----------
    prices : pd.Series
        Close prices.
    signal : pd.Series
        Input signal (can be binary or fractional — non-zero = in trade).
    stop_pct : float
        Trailing stop percentage (0.05 = 5%).
    cooldown : int
        Minimum number of bars to stay flat after a stop-out.

    Returns
    -------
    pd.Series
        Modified signal with stop-outs and cooldowns applied.
    """
    out = signal.copy().astype(float)
    peak = np.nan
    bars_since_stop = cooldown  # start "ready to trade"

    for i in range(len(out)):
        if bars_since_stop < cooldown:
            out.iloc[i] = 0.0
            bars_since_stop += 1
            continue

        if out.iloc[i] != 0:
            price = prices.iloc[i]
            if np.isnan(peak) or price > peak:
                peak = price
            if price < peak * (1 - stop_pct):
                # stop triggered
                out.iloc[i] = 0.0
                peak = np.nan
                bars_since_stop = 0
        else:
            peak = np.nan

    out.name = "signal"
    return out


# ---------------------------------------------------------------------------
# 5.  Full enhanced strategy builder
# ---------------------------------------------------------------------------

def build_enhanced_strategy(
    forecasts: dict[str, pd.Series],
    prices: pd.Series,
    features: pd.DataFrame,
    *,
    min_agreement: float = 0.5,
    vol_window: int = 21,
    target_vol: float = 0.15,
    max_position: float = 1.0,
    rsi_upper: float = 70.0,
    stop_pct: float = 0.05,
    cooldown: int = 3,
) -> pd.Series:
    """Compose the full enhanced strategy pipeline.

    1. Ensemble voting across models
    2. RSI + Bollinger filter
    3. Trailing stop-loss + cooldown
    4. Volatility position sizing

    Returns a position-size series ready for ``backtest_enhanced()``.
    """
    # Step 1 — ensemble consensus
    raw_signal = ensemble_signal(forecasts, prices, min_agreement=min_agreement)

    # Step 2 — RSI / Bollinger filter
    filtered_signal = rsi_bollinger_filter(raw_signal, features, rsi_upper=rsi_upper)

    # Step 3 — trailing stop-loss + cooldown
    stopped_signal = apply_trailing_stop(
        prices, filtered_signal, stop_pct=stop_pct, cooldown=cooldown,
    )

    # Step 4 — volatility-scaled sizing
    position = volatility_position_sizing(
        stopped_signal,
        prices,
        vol_window=vol_window,
        target_vol=target_vol,
        max_position=max_position,
    )

    return position
