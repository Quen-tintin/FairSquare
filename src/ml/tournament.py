"""Model tournament — LinearRegression / GAM / LightGBM."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s, f, l
import lightgbm as lgb

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error, ignoring zero-valued targets.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.

    Returns:
        float: MAPE in percent (e.g. ``15.3`` means 15.3%).
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a standard regression scorecard.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.

    Returns:
        dict: Keys ``MAE``, ``RMSE``, ``R2``, ``MAPE_%``.
    """
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_%": round(_mape(y_true, y_pred), 2),
    }


class ModelTournament:
    """Trains and evaluates 3 models on the same train/test split."""

    def __init__(self) -> None:
        self.models: dict = {
            "LinearRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]),
            "GAM": LinearGAM(
                s(0) + s(1) + f(2) + l(3) + l(4) + f(5) + s(6),
                max_iter=100,
            ),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
            ),
        }
        self.fitted: dict = {}
        self.preds: dict = {}

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        results = []
        for name, model in self.models.items():
            logger.info("Training %s ...", name)
            # pygam expects numpy arrays
            Xtr = X_train.values if name == "GAM" else X_train
            Xte = X_test.values if name == "GAM" else X_test
            model.fit(Xtr, y_train)
            self.fitted[name] = model
            y_pred = model.predict(Xte)
            self.preds[name] = y_pred
            m = _eval(y_test.values, y_pred)
            m["model"] = name
            results.append(m)
            logger.info("%s — MAE=%.1f | RMSE=%.1f | R2=%.4f | MAPE=%.2f%%",
                        name, m["MAE"], m["RMSE"], m["R2"], m["MAPE_%"])
        return pd.DataFrame(results).set_index("model")
