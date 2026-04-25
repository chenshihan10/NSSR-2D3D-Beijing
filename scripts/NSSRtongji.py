import os
import re
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec

# --- 1. 路径配置 ---
ROOT = Path(r"F:\project2025\wulifanyan")
ALIGNED_DIR = ROOT / "XGBoost" / "input" / "NSSR"
OUT_DIR = ROOT / "figures_final_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASON_MAP = {"20230116": "Winter", "20230305": "Spring", "20230828": "Summer", "20231124": "Autumn"}
SEASON_ORDER = ["Summer", "Spring", "Autumn", "Winter"]
COLORS = ["#00008B", "#1E90FF", "#00CED1", "#FFFF00", "#FFA500", "#FF0000"]

def draw_comprehensive_panel():
    # 1. 加载数据
    files = sorted(ALIGNED_DIR.glob("NSSR_Final_Inversion_*.tif"))
    data_dict = {}
    
    for f in files:
        m = re.search(r"(\d{8})", f.name)
        if m and m.group(1) in SEASON_MAP:
            with rasterio.open(f) as src:
                arr = src.read(1).astype(np.float32)
                arr[arr <= 0] = np.nan
                data_dict[SEASON_MAP[m.group(1)]] = arr

    # 2. 绘图布局 (2x2 结构，每个子图内包含地图和直方图)
    fig = plt.figure(figsize=(20, 14), dpi=200)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

    for i, season in enumerate(SEASON_ORDER):
        if season not in data_dict: continue
        arr = data_dict[season]
        v_valid = arr[np.isfinite(arr)]
        
        # 子区域划分：[地图(70%) | 直方图(30%)]
        sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i], wspace=0.2, width_ratios=[1.5, 0.5])
        
        # --- A. 空间分布图 (左) ---
        ax_map = fig.add_subplot(sub_gs[0])
        bounds = np.linspace(np.percentile(v_valid, 2), np.percentile(v_valid, 98), 7)
        norm = BoundaryNorm(bounds, len(COLORS))
        im = ax_map.imshow(arr, cmap=ListedColormap(COLORS), norm=norm)
        ax_map.set_title(f"{season} NSSR Map", fontsize=14, fontweight='bold')
        ax_map.axis('off')
        plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.05, label="W/m²")

        # --- B. 频率直方图 (右) ---
        ax_hist = fig.add_subplot(sub_gs[1])
        ax_hist.hist(v_valid, bins=30, color='#2E8B57', edgecolor='white')
        ax_hist.set_title("Distribution", fontsize=12)
        ax_hist.set_xlabel("NSSR")
        
        # 统计信息框
        total_stats = f"AVG: {np.mean(v_valid):.1f}\nSTD: {np.std(v_valid):.1f}"
        ax_hist.text(0.95, 0.95, total_stats, transform=ax_hist.transAxes, 
                     verticalalignment='top', horizontalalignment='right', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.suptitle("Seasonal NSSR Spatial and Frequency Analysis in Beijing", fontsize=24, fontweight='bold', y=0.96)
    plt.savefig(OUT_DIR / "NSSR_Map_and_Hist_Only3.png", bbox_inches='tight')
    print("分析组图（已移除分类统计表）已生成。")

if __name__ == "__main__":
    draw_comprehensive_panel()