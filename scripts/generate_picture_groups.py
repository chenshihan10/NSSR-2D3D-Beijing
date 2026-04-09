import os
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(r"F:\project2025\wulifanyan")
OUT_DIR = ROOT / "output" / "picture"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_raster(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    return arr


def robust_percentile(arr, p):
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return np.nan
    return float(np.nanpercentile(v, p))


def crop_to_valid(arr, pad=8):
    m = np.isfinite(arr)
    if not np.any(m):
        return arr
    ys, xs = np.where(m)
    r0 = max(int(ys.min()) - pad, 0)
    r1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    c0 = max(int(xs.min()) - pad, 0)
    c1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def find_first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def plot_nssr_clip_group():
    clips = {
        "Winter": ROOT / "output" / "clip" / "NSSR_Clip_20230116.tif",
        "Spring": ROOT / "output" / "clip" / "NSSR_Clip_20230305.tif",
        "Summer": ROOT / "output" / "clip" / "NSSR_Clip_20230828.tif",
        "Autumn": ROOT / "output" / "clip" / "NSSR_Clip_20231124.tif",
    }
    missing = [str(p) for p in clips.values() if not p.exists()]
    if missing:
        print("Skip plot_nssr_clip_group (missing clip rasters).")
        return
    arrays = {k: read_raster(v) for k, v in clips.items()}
    order = ["Winter", "Spring", "Summer", "Autumn"]

    mins = []
    means = []
    maxs = []
    for season in order:
        vals = arrays[season]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        mins.append(float(np.min(vals)))
        means.append(float(np.mean(vals)))
        maxs.append(float(np.max(vals)))

    x = np.arange(len(order))
    w = 0.24
    fig, ax = plt.subplots(figsize=(12, 7), dpi=240)
    b1 = ax.bar(x - w, mins, width=w, color="#5DA5DA", label="Min")
    b2 = ax.bar(x, means, width=w, color="#60BD68", label="Mean")
    b3 = ax.bar(x + w, maxs, width=w, color="#F17CB0", label="Max")

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("NSSR (W/m²)")
    ax.set_title("Seasonal NSSR Statistics (Min / Mean / Max)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 3,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.savefig(OUT_DIR / "seasonal_nssr_bar_stats.png", bbox_inches="tight")
    plt.close(fig)


def plot_nssr_inversion_group():
    rasters = {
        "Winter": ROOT / "output" / "NSSR" / "NSSR_Final_Inversion_Winter_20230116.tif",
        "Spring": ROOT / "output" / "NSSR" / "NSSR_Final_Inversion_Spring_20230305.tif",
        "Summer": ROOT / "output" / "NSSR" / "NSSR_Final_Inversion_Summer_20230828.tif",
        "Autumn": ROOT / "output" / "NSSR" / "NSSR_Final_Inversion_Autumn_20231124.tif",
    }
    for season, path in rasters.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing NSSR inversion raster for {season}: {path}")

    arrays = {k: read_raster(v) for k, v in rasters.items()}
    all_vals = np.concatenate([a[np.isfinite(a)] for a in arrays.values() if np.isfinite(a).any()])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=240)
    order = ["Winter", "Spring", "Summer", "Autumn"]
    cmap = "viridis"
    for ax, season in zip(axes.flat, order):
        arr = arrays[season]
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(season, fontsize=12, pad=6)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("NSSR (W/m²)", fontsize=9)

    fig.suptitle("Seasonal NSSR Inversion Group", fontsize=15, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "seasonal_nssr_inversion_group.png", bbox_inches="tight")
    plt.close(fig)


def plot_morph_group():
    bas_path = find_first_existing([
        ROOT / "Clipped" / "BSA_30m.tif",
        ROOT / "Clipped" / "BAS_30m.tif",
    ])
    mbh_path = ROOT / "Clipped" / "MBH_30m.tif"
    far_path = find_first_existing([
        ROOT / "Clipped" / "FAS_30m.tif",
        ROOT / "Clipped" / "FAR_30m.tif",
    ])
    svf_path = find_first_existing([
        ROOT / "定量归因分析" / "rasters" / "SVF_16dir_30m.tif",
        ROOT / "input" / "svfdata" / "SVF50_30m.tif",
    ])

    missing = [
        ("BSA/BAS", bas_path),
        ("MBH", mbh_path if mbh_path.exists() else None),
        ("FAR", far_path),
        ("SVF30m", svf_path),
    ]
    for name, p in missing:
        if p is None:
            raise FileNotFoundError(f"Missing raster for {name}")

    data = {
        "BSA": read_raster(bas_path),
        "MBH": read_raster(mbh_path),
        "FAR": read_raster(far_path),
        "SVF30m": read_raster(svf_path),
    }
    # Make SVF panel visually larger by cropping to valid data extent.
    data["SVF30m"] = crop_to_valid(data["SVF30m"], pad=10)

    cmaps = {
        "BSA": "magma",
        "MBH": "viridis",
        "FAR": "plasma",
        "SVF30m": "cividis",
    }

    fig = plt.figure(figsize=(14, 10), dpi=240)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 0.05, 1, 0.05], wspace=0.12, hspace=0.14)

    # For each map keep its own right-side colorbar to avoid overlap.
    keys = ["BSA", "MBH", "FAR", "SVF30m"]
    positions = [(0, 0, 0, 1), (0, 2, 0, 3), (1, 0, 1, 1), (1, 2, 1, 3)]

    for k, (r, c_map, r_cb, c_cb) in zip(keys, positions):
        ax = fig.add_subplot(gs[r, c_map])
        cax = fig.add_subplot(gs[r_cb, c_cb])
        arr = data[k]
        vmin = robust_percentile(arr, 2)
        vmax = robust_percentile(arr, 98)
        im = ax.imshow(arr, cmap=cmaps[k], vmin=vmin, vmax=vmax)
        ax.set_title(k, fontsize=11, pad=6)
        ax.axis("off")
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(k, fontsize=9)

    fig.suptitle("BSA, MBH, FAR and SVF30m Group", fontsize=14, y=0.985)
    fig.savefig(OUT_DIR / "BSA_MBH_FAR_SVF30m_group.png", bbox_inches="tight")
    plt.close(fig)


def plot_3dmorph_8factor_group():
    """8-factor 3D morphology group plot (2x4), each with its own scale."""
    tif = ROOT / "output" / "3dmorph" / "Morphology_30m_with_SVF.tif"
    if not tif.exists():
        raise FileNotFoundError(f"Missing morphology raster: {tif}")

    names = ["MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI", "SVF"]
    # Soft, non-dark sequential palettes.
    cmaps = {
        "MBH": "YlGnBu",
        "BU": "YlOrBr",
        "FAR": "PuBuGn",
        "MBV": "BuPu",
        "SCD": "GnBu",
        "BSA": "YlGn",
        "BSI": "PuBu",
        "SVF": "Greens",
    }

    with rasterio.open(tif) as ds:
        arrays = {}
        for i, name in enumerate(names, start=1):
            arr = ds.read(i).astype(np.float32)
            nodata = ds.nodata
            if nodata is not None and np.isfinite(nodata):
                arr[arr == nodata] = np.nan
            arrays[name] = arr

    fig = plt.figure(figsize=(18, 8), dpi=240)
    gs = GridSpec(2, 8, figure=fig, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05], wspace=0.18, hspace=0.18)

    positions = [
        ("MBH", (0, 0), (0, 1)),
        ("BU", (0, 2), (0, 3)),
        ("FAR", (0, 4), (0, 5)),
        ("MBV", (0, 6), (0, 7)),
        ("SCD", (1, 0), (1, 1)),
        ("BSA", (1, 2), (1, 3)),
        ("BSI", (1, 4), (1, 5)),
        ("SVF", (1, 6), (1, 7)),
    ]

    for key, (r_map, c_map), (r_cb, c_cb) in positions:
        ax = fig.add_subplot(gs[r_map, c_map])
        cax = fig.add_subplot(gs[r_cb, c_cb])
        arr = arrays[key]
        vmin = robust_percentile(arr, 2)
        vmax = robust_percentile(arr, 98)
        im = ax.imshow(arr, cmap=cmaps[key], vmin=vmin, vmax=vmax)
        ax.set_title(key, fontsize=11, pad=6)
        ax.axis("off")
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(key, fontsize=9)

    fig.suptitle("3D Morphology Factors (30m) - 8 Panel Group", fontsize=14, y=0.985)
    fig.savefig(OUT_DIR / "morphology_8factor_group.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_nssr_clip_group()
    plot_nssr_inversion_group()
    plot_morph_group()
    plot_3dmorph_8factor_group()
    print("Saved:", OUT_DIR)
