#!/usr/bin/python3

from .db import Database, FakeDatabase

import numpy as np
from catboost import CatBoostRegressor


class Estimator:
    """
    CatBoost model interface for sequential forecasting with SlidingWindow as data storage.
    """
    
    def __init__(self, size=20, refit_threshold=25000):
        self.model = CatBoostRegressor()
        self.data = FakeDatabase()
        self.latest_ema = None
        self.size = size
        self.refit_threshold = refit_threshold
    
    def load_model(self, path: str) -> None:
        self.model.load_model(path)
    
    def update(self, e_mu, e_mu_ema):
        self.data.update(e_mu)
        self.latest_ema = e_mu_ema

    def refit(self):
        sample = self.data.get_latest_rows(self.refit_threshold)
        model = CatBoostRegressor()
        pivoted = np.lib.stride_tricks.sliding_window_view(
            sample, self.size + 1, axis=0
        )
        X, y = pivoted[:, :-1], pivoted[:, -1]
        train = int(self.refit_threshold * 0.8)
        X_train, X_val = X[:train], X[train:]
        y_train, y_val = y[:train], y[train:]
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        self.model = model
    
    def predict(self) -> float | None:
        if self.data.is_full():
            features = self.data.get_latest_rows(self.size)
            return self.model.predict(features)
        else:
            return self.latest_ema

    def close(self):
        self.data.close()
