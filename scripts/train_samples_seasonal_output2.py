from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


ROOT = Path(r"F:\project2025\wulifanyan\XGBoost")
OUT_ROOT = ROOT / "output2"
TRAIN_DIR = OUT_ROOT / "train"
MODEL_DIR = OUT_ROOT / "model"
FIG_DIR = MODEL_DIR / "figures"
METRIC_DIR = MODEL_DIR / "metrics"

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
FEATURES = ["MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI", "SVF"]
TARGET = "NSSR"
N_FOLDS = 5
BLOCK_SIZE_M = 300.0


def season_csv(season: str) -> Path:
    return TRAIN_DIR / f"Training_Samples_{season}_Output2.csv"


def make_spatial_groups(df: pd.DataFrame, block_size_m: float = BLOCK_SIZE_M) -> np.ndarray:
    x_block = np.floor((df["X"].to_numpy() - df["X"].min()) / block_size_m).astype(np.int32)
    y_block = np.floor((df["Y"].to_numpy() - df["Y"].min()) / block_size_m).astype(np.int32)
    groups = y_block * (x_block.max() + 1) + x_block
    return groups.astype(np.int32)


def build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def plot_feature_importance(model: XGBRegressor, season: str, out_path: Path) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    plt.figure(figsize=(7, 4.6), dpi=160)
    plt.barh(imp["feature"], imp["importance"], color="#0f766e")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Feature Importance - {season}")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return imp


def plot_pred_vs_obs(y_true: np.ndarray, y_pred: np.ndarray, season: str, out_path: Path) -> None:
    plt.figure(figsize=(5.8, 5.8), dpi=160)
    plt.scatter(y_true, y_pred, s=2, alpha=0.16, c="#1d4ed8", edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.0)
    plt.xlabel("Observed NSSR")
    plt.ylabel("Predicted NSSR")
    plt.title(f"Pred vs Obs - {season}")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def ensure_dirs() -> None:
    for p in [OUT_ROOT, TRAIN_DIR, MODEL_DIR, FIG_DIR, METRIC_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    rows = []

    for season in SEASONS:
        path = season_csv(season)
        df = pd.read_csv(path, usecols=FEATURES + [TARGET, "X", "Y"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

        X = df[FEATURES].astype(np.float32)
        y = df[TARGET].astype(np.float32)
        groups = make_spatial_groups(df)

        fold_stats = []
        gkf = GroupKFold(n_splits=N_FOLDS)
        hold_pred = hold_true = None
        for fold_id, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
            model = build_model()
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            yt = y.iloc[te].to_numpy()

            fold_stats.append(
                {
                    "Season": season,
                    "Fold": fold_id,
                    "Rows": int(len(te)),
                    "R2": float(r2_score(yt, pred)),
                    "MAE": float(mean_absolute_error(yt, pred)),
                    "RMSE": float(np.sqrt(mean_squared_error(yt, pred))),
                }
            )
            if fold_id == 1:
                hold_true, hold_pred = yt, pred

        fold_df = pd.DataFrame(fold_stats)
        fold_df.to_csv(METRIC_DIR / f"seasonal_cv_folds_{season.lower()}_output2.csv", index=False, encoding="utf-8-sig")

        final_model = build_model()
        final_model.fit(X, y)
        imp = plot_feature_importance(final_model, season, FIG_DIR / f"feature_importance_{season.lower()}_output2.png")
        imp.to_csv(METRIC_DIR / f"feature_importance_{season.lower()}_output2.csv", index=False, encoding="utf-8-sig")

        if hold_true is not None and hold_pred is not None:
            plot_pred_vs_obs(hold_true, hold_pred, season, FIG_DIR / f"pred_vs_obs_{season.lower()}_output2.png")

        row = {
            "Season": season,
            "Rows": int(len(df)),
            "R2_CV_Mean": float(fold_df["R2"].mean()),
            "R2_CV_Std": float(fold_df["R2"].std(ddof=0)),
            "MAE_CV_Mean": float(fold_df["MAE"].mean()),
            "RMSE_CV_Mean": float(fold_df["RMSE"].mean()),
            "Features": ",".join(FEATURES),
            "GroupMethod": f"GroupKFold_{N_FOLDS}_block_{int(BLOCK_SIZE_M)}m",
        }
        rows.append(row)
        print(f"[{season}] rows={row['Rows']:,}, R2={row['R2_CV_Mean']:.4f} ± {row['R2_CV_Std']:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(METRIC_DIR / "seasonal_model_metrics_output2.csv", index=False, encoding="utf-8-sig")
    (METRIC_DIR / "seasonal_model_metrics_output2.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSaved summary:", METRIC_DIR / "seasonal_model_metrics_output2.csv")


if __name__ == "__main__":
    main()

