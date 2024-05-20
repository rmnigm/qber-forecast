#!/usr/bin/python3

from db import Database

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class EMACompositeModel(RegressorMixin, BaseEstimator):
    def __init__(self,
                 alpha: float = 1/3,
                 ) -> None:
        super().__init__()
        self.linear = None
        self.alpha = alpha
        self.boost = LGBMRegressor(verbose=-1)
        
    def fit(self, X, y):
        window_size = X.shape[1]
        self.linear = np.array(
            [self.alpha * (1 - self.alpha) ** i for i in range(window_size)]
        )[::-1]
        predictions = X @ self.linear
        diff = y - predictions
        self.boost.fit(X, diff)
        return self

    def predict(self, X):
        return X @ self.linear + self.boost.predict(X)

    def save(self, path_prefix: str):
        linear_filename = path_prefix + '_linear_model.joblib'
        joblib.dump(self.linear, open(linear_filename, "wb"))
        boost_filename = path_prefix + '_boost_model.joblib'
        joblib.dump(self.boost, open(boost_filename, "wb"))

    def load(self, path_prefix: str):
        linear_filename = path_prefix + '_linear_model.joblib'
        self.linear = joblib.load(linear_filename)
        self.alpha = self.linear[-1]
        boost_filename = path_prefix + '_boost_model.joblib'
        self.boost = joblib.load(boost_filename)


class Estimator:
    """
    CatBoost model interface for sequential forecasting with SlidingWindow as data storage.
    """
    
    def __init__(self, size=20, refit_threshold=20000):
        self.model = EMACompositeModel()
        self.data = Database(db_file="qber.db")
        self.latest_ema = None
        self.size = size
        self.current_sample_len = 0
        self.refit_threshold = refit_threshold
    
    def load_model(self, path_prefix: str) -> None:
        """
        Loads model from file.
        :param path_prefix: str, path prefix to model files
        :return: None
        """
        self.model.load(path_prefix)
    
    def update(self, e_mu, e_mu_ema):
        """
        Updates SlidingWindow with new record, see SlidingWindow.update
        """
        self.data.update(e_mu)
        self.latest_ema = e_mu_ema
        self.current_sample_len += 1
        if self.current_sample_len >= self.refit_threshold:
            sample = self.data.get_latest_rows(self.refit_threshold)
            self.refit(sample)

    def refit(self, sample: np.array):
        model = EMACompositeModel()
        pivoted = np.lib.stride_tricks.sliding_window_view(
            sample, self.size + 1, axis=0
        )
        X, y = pivoted[:, :-1], pivoted[:, -1]
        model.fit(X, y)
        self.model = model
    
    def predict(self) -> float | None:
        """
        Forecasts the new eMu value:
        - if sliding window if full, then calculate the features and pass them to model, return model output
        - if sliding window is not full, but there is at least one record, return latest exponentially averaged value
        - else return None
        :return: float | None, resulting prediction for current timepoint
        """
        if self.data.is_full():
            features = self.data.get_latest_rows(self.size)
            return self.model.predict(features)
        else:
            return self.latest_ema

    def close(self):
        self.data.close()
