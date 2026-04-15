from __future__ import annotations

"""
SHAP 特征重要性分析（对标 Chen et al. 2024）

自变量：8 个城市空间参数（USP）= MBH, BU, FAR, MBV, SCD, BSA, BSI, SVF
因变量：NSSR（地表短波净辐射）

调参与 train_samples_seasonal_output2_tuned.py 一致：
  - 搜索范围：learning_rate∈[0.03,0.05,0.08], max_depth∈[5,6,7,8], min_child_weight∈[1,3,5]
  - GridSearchCV 在训练集上以 KFold(5, shuffle=True, random_state=42) 搜索最优参数
  - 固定参数：n_estimators=2000, gamma=0, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0

数据划分：随机 80/20 分割（train_size=0.8, test_size=0.2, random_state=42）
SHAP 样本：从测试集中随机抽样 min(2000, len(X_test)) 个样本进行特征贡献分析

输出：
  - SHAP 重要性排序（mean absolute SHAP value）
  - 特征依赖图（重点特征：MBH、SVF、BU、FAR）
  - 阈值检测（max gradient slope 方法）
  - 调参参数与汇总表
"""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBRegressor

ROOT = Path(r"F:\project2025\wulifanyan")
TRAIN_DIR = ROOT / "XGBoost" / "output2" / "train"
OUT_DIR = ROOT / "shap"
FIG_DIR = OUT_DIR / "figures"
MET_DIR = OUT_DIR / "metrics"

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
TARGET = "NSSR"
FEATURES = ["MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI", "SVF"]
FOCUS_FEATURES = ["MBH", "SVF", "BU", "FAR"]

GPU_PARAMS = {
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "gpu_id": 0,
    "n_jobs": 1,
}

TEST_FRACTION = 0.2
RANDOM_STATE = 42


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)


def csv_for_season(season: str) -> Path:
    return TRAIN_DIR / f"Training_Samples_{season}_Output2.csv"


def safe_fit(model: XGBRegressor, X: pd.DataFrame, y: pd.Series, **kwargs) -> XGBRegressor:
    try:
        return model.fit(X, y, **kwargs)
    except Exception as exc:
        if model.get_params().get("tree_method") == "gpu_hist":
            print("GPU 训练失败，回退至 CPU hist 模式：", exc)
            model.set_params(tree_method="hist", predictor="auto", n_jobs=-1)
            return model.fit(X, y, **kwargs)
        raise


def build_model(use_gpu: bool = False, gpu_id: int = 0, **kwargs) -> XGBRegressor:
    params = {
        "n_estimators": 2000,  # 与 tuned 脚本一致
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    if use_gpu:
        params.update(GPU_PARAMS)
        params["gpu_id"] = gpu_id
    params.update(kwargs)
    return XGBRegressor(**params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP seasonal analysis with optional GPU acceleration.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="If supported by the installed xgboost, use GPU acceleration for final model fitting.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device id to use when --use-gpu is enabled.",
    )
    return parser.parse_args()


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """与 output2_tuned 一致：仅搜索 learning_rate、max_depth、min_child_weight。"""
    max_rows = 160000
    if len(X_train) > max_rows:
        idx_pos = np.random.RandomState(42).choice(len(X_train), size=max_rows, replace=False)
        X_tune = X_train.iloc[idx_pos]
        y_tune = y_train.iloc[idx_pos]
    else:
        X_tune = X_train
        y_tune = y_train

    param_grid = {
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [5, 6, 7, 8],
        "min_child_weight": [1, 3, 5],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 与 tuned 脚本保持一致
    grid = GridSearchCV(
        estimator=build_model(),
        param_grid=param_grid,
        scoring="r2",  # 改为 R² 评分，与 tuned 脚本一致
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_tune, y_tune)
    return grid.best_params_


def load_season_df(season: str) -> pd.DataFrame:
    usecols = list(dict.fromkeys(FEATURES + [TARGET, "X", "Y"]))
    df = pd.read_csv(csv_for_season(season), usecols=usecols)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=usecols).copy()
    return df


def random_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = df[FEATURES].astype(np.float32)
    y = df[TARGET].astype(np.float32)
    return train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )


def save_summary_plots(shap_values: np.ndarray, X_sample: pd.DataFrame, season: str) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    imp.to_csv(MET_DIR / f"shap_importance_{season.lower()}.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6), dpi=160)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_summary_bar_{season.lower()}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6), dpi=160)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_summary_beeswarm_{season.lower()}.png", bbox_inches="tight")
    plt.close()
    return imp


def estimate_threshold_from_shap(feature_values: np.ndarray, shap_vals: np.ndarray) -> dict:
    order = np.argsort(feature_values)
    x = feature_values[order]
    y = shap_vals[order]
    if len(x) < 20:
        return {"threshold": None, "method": "insufficient_samples"}

    bins = min(40, max(12, len(x) // 30))
    edges = np.linspace(x.min(), x.max(), bins + 1)
    centers = []
    means = []
    for i in range(bins):
        mask = (x >= edges[i]) & (x <= edges[i + 1] if i == bins - 1 else x < edges[i + 1])
        if mask.sum() < 5:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2.0)
        means.append(float(np.mean(y[mask])))

    if len(centers) < 5:
        return {"threshold": None, "method": "insufficient_bins"}

    centers_arr = np.asarray(centers, dtype=np.float32)
    means_arr = np.asarray(means, dtype=np.float32)
    slope = np.gradient(means_arr, centers_arr)
    idx = int(np.argmax(np.abs(slope)))
    return {
        "threshold": float(centers_arr[idx]),
        "method": "max_abs_gradient",
        "slope": float(slope[idx]),
    }


def save_dependence_plot(season: str, feature: str, shap_values: np.ndarray, X_sample: pd.DataFrame) -> dict:
    plt.figure(figsize=(7, 5), dpi=160)
    shap.dependence_plot(feature, shap_values, X_sample, show=False)
    plt.tight_layout()
    out_path = FIG_DIR / f"shap_dependence_{feature.lower()}_{season.lower()}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    feature_idx = list(X_sample.columns).index(feature)
    threshold_info = estimate_threshold_from_shap(
        X_sample[feature].to_numpy(dtype=np.float32),
        shap_values[:, feature_idx].astype(np.float32),
    )
    threshold_info["plot_path"] = str(out_path)
    return threshold_info


def main() -> None:
    args = parse_args()
    ensure_dirs()
    season_rows = []
    threshold_rows = []

    total_seasons = len(SEASONS)
    for i, season in enumerate(SEASONS, start=1):
        # 核心修改：打印进度 [当前/总数]，确保监控脚本能抓取到
        print(f"[{i}/{total_seasons}] Starting SHAP analysis for {season}...", flush=True)
        
        df = load_season_df(season)
        X_train, X_test, y_train, y_test = random_train_test_split(df)
        
        print(f"Tuning model for {season}...", flush=True)
        best_params = tune_model(X_train, y_train)
        
        print(f"Fitting final model for {season}...", flush=True)
        model = build_model(use_gpu=args.use_gpu, gpu_id=args.gpu_id, **best_params)
        safe_fit(model, X_train, y_train)

        print(f"Calculating SHAP values for {season} (this may take a while)...", flush=True)
        sample_n = min(2000, len(X_test))
        X_sample = X_test.sample(sample_n, random_state=42)
        explainer = shap.TreeExplainer(model)
        # 如果样本量大，计算 shap 会很慢，可以在这里加个提示
        shap_values = np.asarray(explainer.shap_values(X_sample), dtype=np.float32)

        print(f"Generating summary plots for {season}...", flush=True)
        imp = save_summary_plots(shap_values, X_sample, season)
        top_n = min(10, len(imp))
        top_features = imp.head(top_n)["feature"].tolist()

        season_rows.append(
            {
                "Season": season,
                "Rows": int(len(df)),
                "SampleSize_SHAP": int(sample_n),
                "Split": f"train_test_{1-TEST_FRACTION:.1f}_{TEST_FRACTION}_random_seed{RANDOM_STATE}",
                "Train_rows": int(len(X_train)),
                "Test_rows": int(len(X_test)),
                "Best_learning_rate": float(best_params["learning_rate"]),
                "Best_max_depth": int(best_params["max_depth"]),
                "Best_min_child_weight": float(best_params["min_child_weight"]),
                "TopFeatures": ",".join(top_features),
            }
        )

        for feature in FOCUS_FEATURES:
            info = save_dependence_plot(season, feature, shap_values, X_sample)
            info["Season"] = season
            info["Feature"] = feature
            threshold_rows.append(info)

    pd.DataFrame(season_rows).to_csv(MET_DIR / "shap_season_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_rows).to_csv(MET_DIR / "shap_thresholds.csv", index=False, encoding="utf-8-sig")
    (MET_DIR / "shap_season_summary.json").write_text(
        json.dumps({"season_summary": season_rows, "thresholds": threshold_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("SHAP analysis pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
