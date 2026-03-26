"""
ML Pipeline — FairSquare
========================
Tournoi de modeles sur DVF Paris 2023 + SHAP XAI
Output : data/outputs/ml/ (JSON metrics + 8 figures PNG)
Lancer : python scripts/run_ml_pipeline.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.features import prepare_features, FEATURE_COLS, FEATURE_LABELS, TARGET
from src.ml.tournament import ModelTournament
from src.ml.xai.shap_explainer import SHAPExplainer

# ── Output dirs ────────────────────────────────────────────────────────────
OUT = Path("data/outputs/ml")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "LinearRegression": "#4C72B0",
    "GAM":              "#DD8452",
    "LightGBM":         "#55A868",
}
sns.set_theme(style="whitegrid", font_scale=1.0)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    print("\n[1/6] Chargement donnees DVF...")
    df = pd.read_parquet("data/processed/dvf_paris_2023_clean.parquet")
    print(f"      {len(df):,} lignes | {df.shape[1]} colonnes")

    # 2. Features
    print("\n[2/6] Preparation des features...")
    X, y = prepare_features(df)
    print(f"      {len(X):,} lignes exploitables | {X.shape[1]} features")
    print(f"      prix_m2 : mean={y.mean():.0f} | std={y.std():.0f} | "
          f"min={y.min():.0f} | max={y.max():.0f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. EDA plots
    print("\n[3/6] Generation des plots EDA...")
    _plot_distribution(y, FIG / "01_prix_m2_distribution.png")
    _plot_correlations(X, y, FIG / "02_feature_correlations.png")
    _plot_prix_by_arr(df, FIG / "03_prix_by_arrondissement.png")
    print("      3 figures EDA sauvegardees")

    # 4. Tournament
    print("\n[4/6] Tournoi de modeles (LinearReg / GAM / LightGBM)...")
    tournament = ModelTournament()
    metrics = tournament.run(X_train, X_test, y_train, y_test)

    print("\n  " + "=" * 55)
    print("       RESULTATS DU TOURNOI DE MODELES")
    print("  " + "=" * 55)
    print(metrics.to_string())

    best = metrics["R2"].idxmax()
    print(f"\n  => Meilleur modele : {best} (R2={metrics.loc[best,'R2']:.4f}, MAE={metrics.loc[best,'MAE']:.0f} EUR/m2)")

    metrics_list = metrics.reset_index().rename(columns={"index": "model"}).to_dict(orient="records")
    (OUT / "metrics.json").write_text(json.dumps(metrics_list, indent=2), encoding="utf-8")

    _plot_model_comparison(metrics, FIG / "04_model_comparison.png")
    _plot_actual_vs_predicted(y_test, tournament.preds, FIG / "05_actual_vs_predicted.png")
    print("      2 figures tournament sauvegardees")

    # 5. SHAP
    print("\n[5/6] Analyse SHAP (LightGBM)...")
    lgbm = tournament.fitted["LightGBM"]
    explainer = SHAPExplainer(lgbm, feature_names=FEATURE_COLS)
    X_shap = X_test.sample(min(500, len(X_test)), random_state=42).reset_index(drop=True)

    explainer.plot_summary(X_shap, FIG / "06_shap_summary.png")
    explainer.plot_bar(X_shap, FIG / "07_shap_importance.png")
    explainer.plot_dependence(X_shap, "surface_reelle_bati", FIG / "08_shap_surface.png")
    print("      3 figures SHAP sauvegardees")

    # 6. Done
    print("\n[6/6] Pipeline complet !")
    print(f"  Figures : {FIG}")
    print(f"  Metrics : {OUT / 'metrics.json'}")
    return metrics, tournament


# ── Plots ──────────────────────────────────────────────────────────────────

def _plot_distribution(y, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[0].axvline(y.mean(),   color="red",    ls="--", lw=1.5, label=f"Moyenne {y.mean():.0f}")
    axes[0].axvline(y.median(), color="orange", ls="--", lw=1.5, label=f"Mediane {y.median():.0f}")
    axes[0].set_xlabel("Prix/m2 (EUR)")
    axes[0].set_ylabel("Transactions")
    axes[0].set_title("Distribution brute")
    axes[0].legend(fontsize=8)

    axes[1].hist(np.log(y), bins=80, color="#55A868", edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("log(Prix/m2)")
    axes[1].set_title("Distribution log-transformee")

    plt.suptitle("DVF Paris 2023 — Prix au m2", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def _plot_correlations(X, y, path):
    data = X.rename(columns=FEATURE_LABELS).copy()
    data["prix_m2"] = y.values
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, center=0, ax=ax, square=True, linewidths=0.5,
    )
    ax.set_title("Correlations — features + prix/m2", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def _plot_prix_by_arr(df, path):
    d = df.copy()
    d["arr"] = (d["code_postal"] % 100).dropna().astype(int)
    d = d.dropna(subset=["code_postal"])
    d["arr"] = (d["code_postal"] % 100).astype(int)
    med = d.groupby("arr")["prix_m2"].median().sort_values(ascending=False)
    cnt = d.groupby("arr")["prix_m2"].count()

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(med)))
    bars = ax.bar(med.index.astype(str), med.values, color=colors, edgecolor="white", lw=0.5)

    for bar, val, c in zip(bars, med.values, cnt[med.index].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{val:,.0f}\n(n={c})", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xlabel("Arrondissement")
    ax.set_ylabel("Prix/m2 median (EUR)")
    ax.set_title("Prix/m2 median par arrondissement — Paris 2023",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def _plot_model_comparison(metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    models = metrics.index.tolist()
    colors = [PALETTE.get(m, "#888") for m in models]

    for ax, col, title, lower_better in zip(
        axes,
        ["MAE", "RMSE", "R2"],
        ["MAE (EUR/m2)\nmoins = mieux", "RMSE (EUR/m2)\nmoins = mieux", "R2\nplus = mieux"],
        [True, True, False],
    ):
        vals = metrics[col].values
        bars = ax.bar(models, vals, color=colors, edgecolor="white", lw=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)

        best_idx = vals.argmin() if lower_better else vals.argmax()
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

        for bar, val in zip(bars, vals):
            label = f"{val:.4f}" if col == "R2" else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    label, ha="center", va="bottom", fontsize=9)

    plt.suptitle("Tournoi de modeles — DVF Paris 2023", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def _plot_actual_vs_predicted(y_test, preds, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        color = PALETTE.get(name, "#888")
        ax.scatter(y_test, y_pred, alpha=0.25, s=4, color=color)
        lims = [
            min(float(y_test.min()), float(y_pred.min())),
            max(float(y_test.max()), float(y_pred.max())),
        ]
        ax.plot(lims, lims, "r--", lw=1.5, label="Parfait")
        ax.set_xlabel("Prix/m2 reel")
        ax.set_ylabel("Prix/m2 predit")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    plt.suptitle("Predit vs Reel — Test set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
