from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    tf = None


def _make_sequences(arr: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback : i])
        y.append(arr[i])
    return np.array(X), np.array(y)


class LSTMForecaster:
    def __init__(
        self,
        lookback: int = 30,
        units: int = 64,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        if tf is None:
            raise ImportError("tensorflow is not installed.")
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.scaler = MinMaxScaler()
        self.model_ = None
        self.history_ = None

    def _build(self) -> "Sequential":
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        m = Sequential(
            [
                Input(shape=(self.lookback, 1)),
                LSTM(self.units, return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.units // 2, return_sequences=False),
                Dropout(self.dropout),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        m.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse", metrics=["mae"])
        return m

    def fit(self, series: pd.Series, validation_split: float = 0.1, verbose: int = 0) -> "LSTMForecaster":
        arr = series.values.reshape(-1, 1).astype(float)
        scaled = self.scaler.fit_transform(arr).flatten()
        X, y = _make_sequences(scaled, self.lookback)
        X = X.reshape(-1, self.lookback, 1)
        self.model_ = self._build()
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.history_ = self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[es],
            verbose=verbose,
        )
        self._last_window_ = scaled[-self.lookback :].copy()
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        window = self._last_window_.copy()
        preds = []
        for _ in range(steps):
            x = window[-self.lookback :].reshape(1, self.lookback, 1)
            yhat = float(self.model_.predict(x, verbose=0)[0, 0])
            preds.append(yhat)
            window = np.append(window, yhat)
        out = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return pd.Series(out, name="forecast")

    def predict_on_test(self, train: pd.Series, test: pd.Series) -> pd.Series:
        full = pd.concat([train, test])
        arr = full.values.reshape(-1, 1).astype(float)
        scaled = self.scaler.transform(arr).flatten()
        start = len(train)
        preds = []
        for i in range(start, len(full)):
            window = scaled[i - self.lookback : i].reshape(1, self.lookback, 1)
            yhat = float(self.model_.predict(window, verbose=0)[0, 0])
            preds.append(yhat)
        out = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return pd.Series(out, index=test.index, name="forecast")
