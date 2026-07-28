from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    def __init__(self, order: Sequence[int] = (5, 1, 0)):
        self.order = tuple(order)
        self.model_ = None
        self.fitted_ = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        self.model_ = ARIMA(series.astype(float), order=self.order)
        self.fitted_ = self.model_.fit()
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self.fitted_ is None:
            raise RuntimeError("Call fit() first.")
        fc = self.fitted_.forecast(steps=steps)
        return pd.Series(np.asarray(fc), name="forecast")

    def summary(self) -> str:
        return str(self.fitted_.summary()) if self.fitted_ is not None else ""

    def walk_forward_forecast(
        self, train: pd.Series, test: pd.Series, refit_every: int = 20,
    ) -> pd.Series:
        """Walk-forward forecast: re-fit on expanding data every *refit_every* steps.

        This prevents the massive drift that occurs when forecasting
        hundreds of steps in one shot from a single ARIMA fit.
        """
        preds = []
        combined = pd.concat([train, test])
        for start in range(0, len(test), refit_every):
            end = min(start + refit_every, len(test))
            train_end = len(train) + start
            history = combined.iloc[:train_end]
            try:
                model = ARIMA(history.astype(float), order=self.order)
                fitted = model.fit()
                chunk = fitted.forecast(steps=end - start)
                preds.extend(chunk.values)
            except Exception:
                # Fallback: repeat last known value
                preds.extend([history.iloc[-1]] * (end - start))
        return pd.Series(preds[: len(test)], index=test.index, name="forecast")


class SARIMAForecaster:
    def __init__(
        self,
        order: Sequence[int] = (1, 1, 1),
        seasonal_order: Sequence[int] = (1, 1, 1, 5),
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.fitted_ = None

    def fit(self, series: pd.Series) -> "SARIMAForecaster":
        model = SARIMAX(
            series.astype(float),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_ = model.fit(disp=False)
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self.fitted_ is None:
            raise RuntimeError("Call fit() first.")
        fc = self.fitted_.forecast(steps=steps)
        return pd.Series(np.asarray(fc), name="forecast")

    def walk_forward_forecast(
        self, train: pd.Series, test: pd.Series, refit_every: int = 20,
    ) -> pd.Series:
        """Walk-forward forecast for SARIMA with periodic re-fitting."""
        preds = []
        combined = pd.concat([train, test])
        for start in range(0, len(test), refit_every):
            end = min(start + refit_every, len(test))
            train_end = len(train) + start
            history = combined.iloc[:train_end]
            try:
                model = SARIMAX(
                    history.astype(float),
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
                chunk = fitted.forecast(steps=end - start)
                preds.extend(chunk.values)
            except Exception:
                preds.extend([history.iloc[-1]] * (end - start))
        return pd.Series(preds[: len(test)], index=test.index, name="forecast")
