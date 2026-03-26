"""SHAP explainability wrapper for LightGBM (TreeExplainer)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class SHAPExplainer:
    """Wraps SHAP TreeExplainer for a fitted LightGBM model."""

    def __init__(self, model, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        return self.explainer.shap_values(X)

    def plot_summary(self, X: pd.DataFrame, output_path: Path) -> None:
        """Beeswarm summary plot (dot style)."""
        vals = self.explainer.shap_values(X)
        plt.figure(figsize=(10, 5))
        shap.summary_plot(
            vals, X, feature_names=self.feature_names, show=False, plot_type="dot"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")

    def plot_bar(self, X: pd.DataFrame, output_path: Path) -> None:
        """Bar chart of mean |SHAP| per feature."""
        vals = self.explainer.shap_values(X)
        plt.figure(figsize=(8, 4))
        shap.summary_plot(
            vals, X, feature_names=self.feature_names, show=False, plot_type="bar"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")

    def plot_dependence(
        self, X: pd.DataFrame, feature: str, output_path: Path
    ) -> None:
        """Dependence plot for a single feature."""
        vals = self.explainer.shap_values(X)
        feat_idx = self.feature_names.index(feature)
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feat_idx,
            vals,
            X.values,
            feature_names=self.feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")
