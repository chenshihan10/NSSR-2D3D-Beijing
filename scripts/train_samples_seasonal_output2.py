from __future__ import annotations

"""
Seasonal XGBoost training for NSSR using eight urban morphology parameters.

This script merges baseline and tuned training modes into one unified workflow.
The training pattern is aligned with the referenced paper, except for the
dependent variable: our study uses NSSR while the paper uses SNR.

Note:
- The paper also uses eight 3D USPs and tunes only eta, max_depth, and
  min_child_weight with 5-fold cross-validation.
- Our current sample generator produces FEATURES = [MBH, BU, FAR, MBV,
  SCD, BSA, BSI, SVF]. The paper uses a similar set but one parameter differs
  (it includes HVC rather than SCD).
"""

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

# train_samples_seasonal_output2.py
BASE_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.02,     # 降低学习率 (从 0.05 降到 0.02)
    "max_depth": 6,            # 减小深度 (从 8 降到 6)，这是防止过拟合最有效的手段
    "min_child_weight": 5,     # 增大叶子节点样本权重 (从 1 加到 5)
    "gamma": 0.1,              # 增加分裂阈值
    "subsample": 0.7,          # 进一步随机化行采样
    "colsample_bytree": 0.7,   # 进一步随机化列采样
    "reg_alpha": 0.1,          # L1 正则化
    "reg_lambda": 1.5,         # L2 正则化
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",     # 保持 hist 模式
    "random_state": 42,
}
GPU_PARAMS = {
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "gpu_id": 0,
    "n_jobs": 1,
}

TUNE_GRID = {
    "learning_rate": [0.03, 0.05, 0.08],
    "max_depth": [5, 6, 7, 8],
    "min_child_weight": [1, 3, 5],
}

TARGET_TRANSFORMS = {
    "none": (lambda y: y, lambda y: y),
    "log1p": (np.log1p, np.expm1),
}


def season_csv(season: str) -> Path:
    return TRAIN_DIR / f"Training_Samples_{season}_Output2.csv"


def safe_fit(model: XGBRegressor, X: pd.DataFrame, y: pd.Series, **kwargs) -> XGBRegressor:
    """
    终极安全训练函数：专为全量训练设计，彻底杜绝死循环。
    """
    # --- 策略：前置检查 ---
    # 如果 kwargs 里没有提供验证集，必须主动关掉早停开关
    if "eval_set" not in kwargs:
        if model.get_params().get("early_stopping_rounds") is not None:
            print(">> [safe_fit] 全量训练模式：检测到未提供验证集，已自动禁用早停。", flush=True)
            model.set_params(early_stopping_rounds=None)

    try:
        # 第一次尝试训练
        return model.fit(X, y, **kwargs)
    except Exception as exc:
        print(f">> [safe_fit] 首次训练失败，原因: {exc}", flush=True)
        
        # --- 策略：强制降级回退 ---
        # 如果报错（无论是硬件还是参数），直接切到最原始、最兼容的 CPU 模式
        print(">> [safe_fit] 正在执行最后的安全回退：强制切换至基础 CPU 模式...", flush=True)
        
        model.set_params(
            tree_method="hist", 
            device="cpu", 
            n_jobs=-1,
            early_stopping_rounds=None, # 双重保险
            predictor="cpu_predictor"
        )
        
        # 【关键】这里直接调用原生 fit，不再调用 safe_fit 本身，从而切断递归链
        return model.fit(X, y, **kwargs)


def build_model(params: dict | None = None, use_gpu: bool = False, gpu_id: int = 0) -> XGBRegressor:
    # 1. 准备基础参数字典
    cfg = dict(BASE_PARAMS)
    if params:
        cfg.update(params)
    
    # 2. 移除与新版 API 冲突或重复的键
    for key in ["early_stopping_rounds", "eval_metric", "tree_method", "device", "gpu_id"]:
        cfg.pop(key, None)

    # 3. 配置硬件加速
    tree_method = "hist"
    device = "cpu"
    if use_gpu:
        tree_method = "hist"  # XGB 2.0+ gpu_hist 已并入 hist
        device = f"cuda:{gpu_id}"

    # 4. 显式通过构造函数参数创建模型
    return XGBRegressor(
        **cfg,
        tree_method=tree_method,
        device=device,
        eval_metric="rmse",          # 必须在这里
        early_stopping_rounds=50      # 必须在这里
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


def plot_pred_vs_obs(
    y_true: np.ndarray, y_pred: np.ndarray, season: str, out_path: Path, *, oof: bool = False
) -> None:
    plt.figure(figsize=(5.8, 5.8), dpi=160)
    plt.scatter(y_true, y_pred, s=2, alpha=0.16, c="#1d4ed8", edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.0)
    plt.xlabel(f"Observed {TARGET}")
    plt.ylabel(f"Predicted {TARGET}")
    suffix = " (OOF)" if oof else ""
    plt.title(f"Pred vs Obs - {season}{suffix}")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def ensure_dirs() -> None:
    for p in [OUT_ROOT, TRAIN_DIR, MODEL_DIR, FIG_DIR, METRIC_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    target_transform: str = "none",
    use_gpu: bool = False,
    gpu_id: int = 0,
) -> tuple[float, float, float]:
    transformer, inverse = TARGET_TRANSFORMS[target_transform]
    y_trans = transformer(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    r2s, maes, rmses = [], [], []
    for tr, te in kf.split(X, y_trans):
        model = build_model(params, use_gpu=use_gpu, gpu_id=gpu_id)
        safe_fit(model, X.iloc[tr], y_trans.iloc[tr], eval_set=[(X.iloc[te], y_trans.iloc[te])])
        pred = model.predict(X.iloc[te])
        if target_transform != "none":
            pred = inverse(pred)
        yt = y.iloc[te].to_numpy()
        r2s.append(float(r2_score(yt, pred)))
        maes.append(float(mean_absolute_error(yt, pred)))
        rmses.append(float(np.sqrt(mean_squared_error(yt, pred))))
    return float(np.mean(r2s)), float(np.mean(maes)), float(np.mean(rmses))


def train_baseline(season: str, use_gpu: bool = False) -> dict:
    df = pd.read_csv(season_csv(season), usecols=FEATURES + [TARGET, "X", "Y"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = df[FEATURES].astype(np.float32)
    y = df[TARGET].astype(np.float32)

    fold_stats = []
    oof_pred = np.full(len(y), np.nan, dtype=np.float64)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # 获取季节索引用于显示总进度
    season_idx = SEASONS.index(season) + 1
    total_seasons = len(SEASONS)

    for fold_id, (tr, te) in enumerate(kf.split(X, y), start=1):
        # 1. 打印详细进度，触发网页进度条小步跳动
        print(f"[{season_idx}/{total_seasons}] Fold {fold_id}/{N_FOLDS}: Training {season}...", flush=True)
        model = build_model(BASE_PARAMS, use_gpu=use_gpu)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[te], y.iloc[te])],
            verbose=100
        )
        
        # 3. 预测与性能对比
        pred_te = model.predict(X.iloc[te])
        oof_pred[te] = pred_te  # 存储 OOF 预测结果以供后续分析
        pred_tr = model.predict(X.iloc[tr])
        actual_te = y.iloc[te] 
        r2_test = float(r2_score(actual_te, pred_te))
        r2_train = float(r2_score(y.iloc[tr], pred_tr))
        print("-" * 30, flush=True)
        print(f">> Fold {fold_id} Metrics: Train R2={r2_train:.4f}, Test R2={r2_test:.4f}", flush=True)
        
        if r2_train - r2_test > 0.15:
            # 这里的关键字要和 HTML 中的 parseProgress 匹配
            print(f"警告：发现严重过拟合倾向！(Gap: {r2_train - r2_test:.4f})", flush=True)
        print("-" * 30, flush=True)

        fold_stats.append({
            "Season": season,
            "Fold": fold_id,
            "Rows": int(len(te)),
            "R2": r2_test,
            "Train_R2": r2_train,
            "MAE": float(mean_absolute_error(actual_te, pred_te)), # 修正后
            "RMSE": float(np.sqrt(mean_squared_error(actual_te, pred_te))), # 修正后
        })

    fold_df = pd.DataFrame(fold_stats)
    fold_df.to_csv(METRIC_DIR / f"seasonal_cv_folds_{season.lower()}_output2_baseline.csv", index=False, encoding="utf-8-sig")
    # Final fit with an internal holdout to surface overfitting risk.
    # This is not a spatially independent validation, but is better than train-only loss.
    print(f"[{season_idx}/{total_seasons}] Starting Final Fit (Holdout 10%)...", flush=True)
    rs = np.random.RandomState(42)
    idx = np.arange(len(X), dtype=np.int64)
    rs.shuffle(idx)
    n_val = max(1, int(0.1 * len(idx)))
    idx_val = idx[:n_val]
    idx_tr = idx[n_val:]

    X_tr = X.iloc[idx_tr]
    y_tr = y.iloc[idx_tr]
    X_val = X.iloc[idx_val]
    y_val = y.iloc[idx_val]

    final_model = build_model(BASE_PARAMS, use_gpu=use_gpu)
    safe_fit(
        final_model,
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # Refit on full data using best_iteration to keep the same early-stopped complexity.
    best_iter = getattr(final_model, "best_iteration", None)
    if best_iter is not None and isinstance(best_iter, int) and best_iter > 0:
        refit_model = build_model(BASE_PARAMS, use_gpu=use_gpu)
        refit_model.set_params(n_estimators=int(best_iter))
        safe_fit(refit_model, X, y, verbose=False)
        refit_model.save_model(str(MODEL_DIR / f"xgb_nssr_{season.lower()}_output2_baseline.json"))
        final_model = refit_model
    else:
        final_model.save_model(str(MODEL_DIR / f"xgb_nssr_{season.lower()}_output2_baseline.json"))
    imp = plot_feature_importance(final_model, season, FIG_DIR / f"feature_importance_{season.lower()}_output2_baseline.png")
    imp.to_csv(METRIC_DIR / f"feature_importance_{season.lower()}_output2_baseline.csv", index=False, encoding="utf-8-sig")
    if np.all(np.isfinite(oof_pred)):
        plot_pred_vs_obs(
            y.to_numpy(),
            oof_pred.astype(np.float32),
            season,
            FIG_DIR / f"pred_vs_obs_{season.lower()}_output2_baseline.png",
            oof=True,
        )

    return {
        "Season": season,
        "Rows": int(len(df)),
        "R2_CV_Mean": float(fold_df["R2"].mean()),
        "R2_CV_Std": float(fold_df["R2"].std(ddof=0)),
        "R2_OOF": float(r2_score(y.to_numpy(), oof_pred)),
        "MAE_CV_Mean": float(fold_df["MAE"].mean()),
        "RMSE_CV_Mean": float(fold_df["RMSE"].mean()),
        "Best_params": BASE_PARAMS,
        "Target_transform": "none",
    }


def tune_season(season: str, target_transform: str = "none", use_gpu: bool = False, gpu_id: int = 0) -> dict:
    df = pd.read_csv(season_csv(season), usecols=FEATURES + [TARGET, "X", "Y"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = df[FEATURES].astype(np.float32)
    y = df[TARGET].astype(np.float32)

    candidates = [dict(zip(TUNE_GRID.keys(), vals)) for vals in product(*TUNE_GRID.values())]
    best = None
    for params in candidates:
        r2, mae, rmse = evaluate_cv(
            X,
            y,
            params,
            target_transform=target_transform,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )
        if best is None or r2 > best["R2_CV_Mean"]:
            best = {
                "Season": season,
                "Rows": int(len(df)),
                "R2_CV_Mean": r2,
                "MAE_CV_Mean": mae,
                "RMSE_CV_Mean": rmse,
                "SplitMethod": f"KFold_{N_FOLDS}_shuffle_seed42",
                **params,
            }

    p_best = {k: best[k] for k in TUNE_GRID.keys()}
    transformer, inverse = TARGET_TRANSFORMS[target_transform]
    y_trans = transformer(y)
    final_model = build_model(p_best, use_gpu=use_gpu, gpu_id=gpu_id)
    safe_fit(final_model, X, y_trans)
    final_model.set_params(early_stopping_rounds=None)
    final_model.save_model(str(MODEL_DIR / f"xgb_nssr_{season.lower()}_output2_tuned.json"))

    with open(METRIC_DIR / f"xgb_nssr_{season.lower()}_output2_tuned.meta.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "season": season,
                "dependent_variable": TARGET,
                "independent_variables": FEATURES,
                "target_transform": target_transform,
                "inverse_transform": "none" if target_transform == "none" else "expm1",
                "tuning_params": list(TUNE_GRID.keys()),
                "fixed_xgb_defaults": {
                    k: BASE_PARAMS[k] for k in ["n_estimators", "gamma", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"]
                },
                "best_params": p_best,
                "cv_metrics": {
                    "R2_CV_Mean": best["R2_CV_Mean"],
                    "MAE_CV_Mean": best["MAE_CV_Mean"],
                    "RMSE_CV_Mean": best["RMSE_CV_Mean"],
                },
                "model_path": str(MODEL_DIR / f"xgb_nssr_{season.lower()}_output2_tuned.json"),
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seasonal XGBoost training for NSSR with baseline and tuned modes.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "tuned", "all"],
        default="all",
        help="Run baseline only, tuned only, or both.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["none", "log1p"],
        default="none",
        help="Target transform for tuning. Article-style training uses raw SNR/NSSR, so default is none.",
    )
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
    ensure_dirs()

    baseline_rows = []
    tuned_rows = []

    total = len(SEASONS)
    for i, season in enumerate(SEASONS, start=1):
        # 必须打印这种格式，监控网页才能识别进度条 [当前/总数]
        print(f"[{i}/{total}] Processing season: {season}", flush=True)
        
        if args.mode in ["baseline", "all"]:
            print(f"Running baseline for {season}...", flush=True)
            baseline_rows.append(train_baseline(season, use_gpu=args.use_gpu))
            
        if args.mode in ["tuned", "all"]:
            print(f"Tuning {season}...", flush=True)
            tuned_rows.append(tune_season(season, target_transform=args.target_transform, use_gpu=args.use_gpu))

    print("Finished all tasks.", flush=True)
   

if __name__ == "__main__":
    main()
