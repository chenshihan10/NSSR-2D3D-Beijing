from __future__ import annotations

"""
分季节 XGBoost 调参版（对标 Chen et al. 2024）

自变量：8 个城市空间参数（USP）= MBH, BU, FAR, MBV, SCD, BSA, BSI, SVF
因变量：NSSR（地表短波净辐射；文献称 SNR）

网格搜索范围（仅调这三项）：
  - learning_rate (eta): [0.03, 0.05, 0.08]
  - max_depth: [5, 6, 7, 8]
  - min_child_weight: [1, 3, 5]
  共 3×4×3 = 36 组，用 K 折平均 R² 选最优

固定参数（与基线 train_samples_seasonal_output2.py 保持一致）：
  n_estimators=2000, gamma=0, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0

评价指标：K 折平均 R²（主）、MAE、RMSE（辅助）
划分方式：KFold(n=5, shuffle=True, random_state=42) 与基线一致
目标变换：log1p(NSSR) 训练，expm1(预测) 评估

输出：
  - 四季最优模型（.joblib）
  - 元数据含最优参数与CV指标（.meta.json）
  - 调参网格搜索结果汇总（seasonal_model_metrics_output2_tuned.csv/json）
"""

from itertools import product
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


ROOT = Path(r"F:\project2025\wulifanyan\XGBoost")
OUT_ROOT = ROOT / "output2"
TRAIN_DIR = OUT_ROOT / "train"
OUT_DIR = OUT_ROOT / "model_tuned"
METRIC_DIR = OUT_DIR / "metrics"
MODEL_DIR = OUT_DIR / "saved_models"

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
FEATURES = ["MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI", "SVF"]
TARGET = "NSSR"
N_FOLDS = 5

GPU_PARAMS = {
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "gpu_id": 0,
    "n_jobs": 1,
}

# 仅调节文献明确的三项；其余与常见默认一致并固定
TUNE_GRID = {
    "learning_rate": [0.03, 0.05, 0.08],
    "max_depth": [5, 6, 7, 8],
    "min_child_weight": [1, 3, 5],
}


def season_csv(season: str) -> Path:
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


def build_model(params: dict, use_gpu: bool = False, gpu_id: int = 0) -> XGBRegressor:
    config = {
        "n_estimators": 2000,
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
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
        config.update(GPU_PARAMS)
        config["gpu_id"] = gpu_id
    return XGBRegressor(**config)


def evaluate_cv(
    X: pd.DataFrame, y_log: pd.Series, y_raw: pd.Series, params: dict, use_gpu: bool = False, gpu_id: int = 0
) -> tuple[float, float, float]:
    """返回该组超参下的 K 折平均 R²、MAE、RMSE（原始 NSSR 尺度）。"""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    r2s, maes, rmses = [], [], []
    for tr, te in kf.split(X, y_log):
        model = build_model(params, use_gpu=use_gpu, gpu_id=gpu_id)
        safe_fit(
            model,
            X.iloc[tr],
            y_log.iloc[tr],
            eval_set=[(X.iloc[te], y_log.iloc[te])],
            verbose=False,
        )
        pred_log = model.predict(X.iloc[te])
        pred = np.expm1(pred_log)
        yt = y_raw.iloc[te].to_numpy()
        r2s.append(float(r2_score(yt, pred)))
        maes.append(float(mean_absolute_error(yt, pred)))
        rmses.append(float(np.sqrt(mean_squared_error(yt, pred))))
    return float(np.mean(r2s)), float(np.mean(maes)), float(np.mean(rmses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tuned seasonal XGBoost training with optional GPU acceleration.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="If supported by the installed xgboost, attempt GPU-accelerated training.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device id to use when --use-gpu is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [dict(zip(TUNE_GRID.keys(), vals)) for vals in product(*TUNE_GRID.values())]

    rows = []
    for season in SEASONS:
        df = pd.read_csv(season_csv(season), usecols=FEATURES + [TARGET, "X", "Y"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
        X = df[FEATURES].astype(np.float32)
        y_raw = df[TARGET].astype(np.float32)
        y_log = np.log1p(y_raw)

        best = None
        for p in candidates:
            r2, mae, rmse = evaluate_cv(
                X,
                y_log,
                y_raw,
                p,
                use_gpu=args.use_gpu,
                gpu_id=args.gpu_id,
            )
            if best is None or r2 > best["R2_CV_Mean"]:
                best = {
                    "Season": season,
                    "Rows": int(len(df)),
                    "R2_CV_Mean": r2,
                    "MAE_CV_Mean": mae,
                    "RMSE_CV_Mean": rmse,
                    "SplitMethod": f"KFold_{N_FOLDS}_shuffle_seed42",
                    **p,
                }
        rows.append(best)

        p_best = {k: best[k] for k in TUNE_GRID.keys()}
        final_model = build_model(p_best, use_gpu=args.use_gpu, gpu_id=args.gpu_id)
        safe_fit(final_model, X, y_log)
        model_path = MODEL_DIR / f"xgb_nssr_{season.lower()}_output2_tuned.joblib"
        joblib.dump(final_model, model_path)
        meta_path = MODEL_DIR / f"xgb_nssr_{season.lower()}_output2_tuned.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "season": season,
                    "dependent_variable": "NSSR",
                    "independent_variables": FEATURES,
                    "target_transform": "log1p(y)",
                    "inverse_transform": "expm1(pred)",
                    "tuning_params": list(TUNE_GRID.keys()),
                    "fixed_xgb_defaults": {
                        "n_estimators": 2000,
                        "gamma": 0.0,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_lambda": 1.0,
                        "reg_alpha": 0.0,
                    },
                    "best_params": p_best,
                    "cv_metrics": {
                        "R2_CV_Mean": best["R2_CV_Mean"],
                        "MAE_CV_Mean": best["MAE_CV_Mean"],
                        "RMSE_CV_Mean": best["RMSE_CV_Mean"],
                    },
                    "model_path": str(model_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[{season}] R2={best['R2_CV_Mean']:.4f} MAE={best['MAE_CV_Mean']:.4f} "
            f"best_eta={p_best['learning_rate']} max_depth={p_best['max_depth']} "
            f"min_child_weight={p_best['min_child_weight']} -> {model_path}"
        )

    out_df = pd.DataFrame(rows)
    out_csv = METRIC_DIR / "seasonal_model_metrics_output2_tuned.csv"
    out_json = METRIC_DIR / "seasonal_model_metrics_output2_tuned.json"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
