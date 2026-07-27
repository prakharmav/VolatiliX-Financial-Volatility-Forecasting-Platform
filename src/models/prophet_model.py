from __future__ import annotations

import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


class ProphetForecaster:
    def __init__(
        self,
        daily_seasonality: bool = False,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
    ):
        if Prophet is None:
            raise ImportError("prophet is not installed. `pip install prophet`")
        self.kwargs = dict(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        self.model_ = None

    def fit(self, series: pd.Series) -> "ProphetForecaster":
        df = pd.DataFrame({"ds": series.index, "y": series.values})
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)
        self.model_ = Prophet(**self.kwargs)
        self.model_.fit(df)
        return self

    def forecast(self, steps: int, freq: str = "B") -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        future = self.model_.make_future_dataframe(periods=steps, freq=freq)
        fc = self.model_.predict(future)
        return fc.set_index("ds")["yhat"].iloc[-steps:].rename("forecast")

    def forecast_full(self, steps: int, freq: str = "B") -> pd.DataFrame:
        future = self.model_.make_future_dataframe(periods=steps, freq=freq)
        return self.model_.predict(future)
