# 二三维城市结构对季节性地表短波净辐射的影响

## 项目简介
本项目聚焦北京市研究区（方形研究区）内，评估二/三维城市结构因子对季节性地表短波净辐射（NSSR）的影响。
当前版本统一采用：
- NSSR 反演公式：`NSSR = DSSR * (1 - Albedo)`
- ERA5 数据源：`input/era52`（优化版，替代旧版 `ERA5`）

## 目录结构
- `data/nssr`：四季 NSSR 反演结果
- `data/landcover`：地表覆盖数据（ESA LC 30m）
- `scripts`：核心脚本
- `results/diwufenlei`：基于地物分类的二维定量评估结果
- `results/xgboost_output2`：最新版 XGBoost 训练与 SHAP 结果
- `results/figures`：关键图件（四季组图、统计图等）

## 外部数据说明（不随仓库上传）
以下原始数据体量较大，不在仓库内提交：
1. Albedo 数据（原路径：`input/Albedo`）
2. ERA5 优化数据（原路径：`input/era52`）

使用时请将上述数据放回对应路径，再运行脚本。

## 核心脚本
- `scripts/fanyan.py`：四季 NSSR 反演（读取 Albedo 与 era52）
- `scripts/diwufenlei.py`：NSSR × Land Cover 二维分区统计与图件输出
- `scripts/build_3dmorph.py`：30m 三维形态因子构建（含 SVF 合并）
- `scripts/create_training_samples_output2.py`：最新样本集生成（output2）
- `scripts/train_samples_seasonal_output2.py`：四季解耦训练与评估

## 关键图件
- `results/figures/seasonal_nssr_inversion_group.png`
- `results/figures/seasonal_nssr_bar_stats.png`
- `results/figures/BSA_vs_SVF_density.png`

## 复现顺序
1. 放置 Albedo 与 era52 原始数据
2. 运行 `scripts/fanyan.py`
3. 运行 `scripts/diwufenlei.py`
4. 运行 `scripts/create_training_samples_output2.py`
5. 运行 `scripts/train_samples_seasonal_output2.py`
