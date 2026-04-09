from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


ROOT = Path(r"F:\project2025\wulifanyan")
NSSR_DIR = ROOT / "output" / "NSSR"
LC_PATH = ROOT / "input" / "diwufenlei" / "Beijing_ESA_LC_2023_30m.tif"
OUT_DIR = ROOT / "output" / "diwufenlei"
FIG_DIR = OUT_DIR / "figures"

SEASON_FILES = {
    "Winter": NSSR_DIR / "NSSR_Final_Inversion_Winter_20230116.tif",
    "Spring": NSSR_DIR / "NSSR_Final_Inversion_Spring_20230305.tif",
    "Summer": NSSR_DIR / "NSSR_Final_Inversion_Summer_20230828.tif",
    "Autumn": NSSR_DIR / "NSSR_Final_Inversion_Autumn_20231124.tif",
}

# ESA WorldCover classes (2023):
# 10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 50 Built-up,
# 60 Bare / sparse vegetation, 70 Snow and ice, 80 Permanent water bodies, ...
LC_GROUPS = {
    "Forest": [10],
    "Shrubland": [20],
    "Grassland": [30],
    "Cropland": [40],
    "Buildings": [50],
    "Bareland": [60],
    "SnowIce": [70],
    "Water": [80],
}
EXCLUDE_CLASSES: set[int] = set()


def _read_nssr(path: Path) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            arr = np.where(arr == nodata, np.nan, arr)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr, src.transform, src.crs


def _read_lc_aligned(target_shape: tuple[int, int], target_transform, target_crs) -> np.ndarray:
    with rasterio.open(LC_PATH) as src:
        lc_aligned = np.full(target_shape, 0, dtype=np.int16)
        reproject(
            source=rasterio.band(src, 1),
            destination=lc_aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
    return lc_aligned


def _season_stats(season: str, nssr_path: Path) -> pd.DataFrame:
    nssr, transform, crs = _read_nssr(nssr_path)
    lc = _read_lc_aligned(nssr.shape, transform, crs)

    valid = np.isfinite(nssr) & (nssr > 0) & (lc > 0)
    if EXCLUDE_CLASSES:
        valid &= ~np.isin(lc, list(EXCLUDE_CLASSES))

    rows: list[dict[str, float | int | str]] = []
    for lc_name, codes in LC_GROUPS.items():
        m = valid & np.isin(lc, codes)
        vals = nssr[m]
        if vals.size == 0:
            rows.append(
                {
                    "Season": season,
                    "LC_Class": lc_name,
                    "Mean_NSSR": np.nan,
                    "StdDev_NSSR": np.nan,
                    "Pixel_Count": 0,
                }
            )
            continue
        rows.append(
                {
                    "Season": season,
                    "LC_Class": lc_name,
                    "Mean_NSSR": float(np.nanmean(vals)),
                    "StdDev_NSSR": float(np.nanstd(vals)),
                    "P10_NSSR": float(np.nanpercentile(vals, 10)),
                    "P50_NSSR": float(np.nanpercentile(vals, 50)),
                    "P90_NSSR": float(np.nanpercentile(vals, 90)),
                    "IQR_NSSR": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
                    "Spread_P90_P10": float(np.nanpercentile(vals, 90) - np.nanpercentile(vals, 10)),
                    "Pixel_Count": int(vals.size),
                }
            )
    return pd.DataFrame(rows)


def _build_delta_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, float | str]] = []
    for season in stats_df["Season"].unique():
        sub = stats_df[stats_df["Season"] == season].set_index("LC_Class")
        built = sub.loc["Buildings", "Mean_NSSR"] if "Buildings" in sub.index else np.nan
        grass = sub.loc["Grassland", "Mean_NSSR"] if "Grassland" in sub.index else np.nan
        forest = sub.loc["Forest", "Mean_NSSR"] if "Forest" in sub.index else np.nan
        natural = np.nanmean([grass, forest])
        out_rows.append(
            {
                "Season": season,
                "Buildings_Mean_NSSR": built,
                "Natural_Mean_NSSR": natural,
                "Delta_NSSR_Buildings_minus_Natural": built - natural if np.isfinite(built) and np.isfinite(natural) else np.nan,
            }
        )
    return pd.DataFrame(out_rows)


def _build_contribution_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for season, sub in stats_df.groupby("Season"):
        tmp = sub.copy()
        tmp["NSSR_Sum"] = tmp["Mean_NSSR"] * tmp["Pixel_Count"]
        total_sum = float(np.nansum(tmp["NSSR_Sum"].to_numpy(dtype=np.float64)))
        total_pix = int(np.nansum(tmp["Pixel_Count"].to_numpy(dtype=np.int64)))

        for _, r in tmp.iterrows():
            cls_sum = float(r["NSSR_Sum"]) if np.isfinite(r["NSSR_Sum"]) else np.nan
            cls_pix = int(r["Pixel_Count"])
            rows.append(
                {
                    "Season": season,
                    "LC_Class": r["LC_Class"],
                    "NSSR_Sum": cls_sum,
                    "NSSR_Share": cls_sum / total_sum if np.isfinite(cls_sum) and total_sum > 0 else np.nan,
                    "Pixel_Share": cls_pix / total_pix if total_pix > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _core_suburb_stats(season: str, nssr_path: Path) -> pd.DataFrame:
    nssr, transform, crs = _read_nssr(nssr_path)
    lc = _read_lc_aligned(nssr.shape, transform, crs)

    valid = np.isfinite(nssr) & (nssr > 0) & (lc > 0)
    if EXCLUDE_CLASSES:
        valid &= ~np.isin(lc, list(EXCLUDE_CLASSES))

    rows_idx, cols_idx = np.indices(nssr.shape)
    xs = transform.c + (cols_idx + 0.5) * transform.a
    ys = transform.f + (rows_idx + 0.5) * transform.e

    xv = xs[valid]
    yv = ys[valid]
    nssr_v = nssr[valid]
    lc_v = lc[valid]
    if nssr_v.size == 0:
        return pd.DataFrame(columns=["Season", "Zone", "LC_Class", "Mean_NSSR", "StdDev_NSSR", "Pixel_Count"])

    x0 = float(np.nanmean(xv))
    y0 = float(np.nanmean(yv))
    dist = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2)
    q_core = float(np.nanpercentile(dist, 40))
    q_suburb = float(np.nanpercentile(dist, 70))

    zones = {
        "Core": dist <= q_core,
        "Suburb": dist >= q_suburb,
    }

    out_rows: list[dict[str, float | int | str]] = []
    for zone_name, zmask in zones.items():
        out_rows.append(
            {
                "Season": season,
                "Zone": zone_name,
                "LC_Class": "All",
                "Mean_NSSR": float(np.nanmean(nssr_v[zmask])) if np.any(zmask) else np.nan,
                "StdDev_NSSR": float(np.nanstd(nssr_v[zmask])) if np.any(zmask) else np.nan,
                "Pixel_Count": int(np.sum(zmask)),
            }
        )
        for lc_name, codes in LC_GROUPS.items():
            cmask = zmask & np.isin(lc_v, codes)
            vals = nssr_v[cmask]
            out_rows.append(
                {
                    "Season": season,
                    "Zone": zone_name,
                    "LC_Class": lc_name,
                    "Mean_NSSR": float(np.nanmean(vals)) if vals.size > 0 else np.nan,
                    "StdDev_NSSR": float(np.nanstd(vals)) if vals.size > 0 else np.nan,
                    "Pixel_Count": int(vals.size),
                }
            )
    return pd.DataFrame(out_rows)


def _season_change_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    pivot = stats_df.pivot(index="LC_Class", columns="Season", values="Mean_NSSR")
    order = ["Winter", "Spring", "Summer", "Autumn"]
    pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])
    rows = []
    for cls, row in pivot.iterrows():
        vals = row.to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            rows.append({"LC_Class": cls, "Seasonal_Min": np.nan, "Seasonal_Max": np.nan, "Amplitude": np.nan, "Mean_Across_Seasons": np.nan, "CV_Across_Seasons": np.nan})
            continue
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        rows.append(
            {
                "LC_Class": cls,
                "Seasonal_Min": float(np.min(vals)),
                "Seasonal_Max": float(np.max(vals)),
                "Amplitude": float(np.max(vals) - np.min(vals)),
                "Mean_Across_Seasons": mean_v,
                "CV_Across_Seasons": std_v / mean_v if mean_v > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("Amplitude", ascending=False)
    return out


def _build_core_suburb_gradient(core_suburb_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in core_suburb_df["Season"].dropna().unique():
        sub = core_suburb_df[(core_suburb_df["Season"] == season) & (core_suburb_df["LC_Class"] == "All")]
        if {"Core", "Suburb"}.issubset(set(sub["Zone"].tolist())):
            core_mean = float(sub[sub["Zone"] == "Core"]["Mean_NSSR"].iloc[0])
            suburb_mean = float(sub[sub["Zone"] == "Suburb"]["Mean_NSSR"].iloc[0])
            rows.append(
                {
                    "Season": season,
                    "Core_Mean_NSSR": core_mean,
                    "Suburb_Mean_NSSR": suburb_mean,
                    "Core_minus_Suburb": core_mean - suburb_mean,
                }
            )
    return pd.DataFrame(rows)


def _plot_grouped_means(stats_df: pd.DataFrame, out_png: Path) -> None:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    lc_order = list(LC_GROUPS.keys())
    pivot = stats_df.pivot(index="LC_Class", columns="Season", values="Mean_NSSR").reindex(lc_order)

    x = np.arange(len(lc_order), dtype=np.float32)
    width = 0.18
    plt.figure(figsize=(13, 6), dpi=180)
    for i, season in enumerate(seasons):
        vals = pivot[season].to_numpy(dtype=np.float32)
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=season)
    plt.xticks(x, lc_order, rotation=25)
    plt.ylabel("Mean NSSR (W/m²)")
    plt.title("Seasonal Mean NSSR by Land Cover Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _plot_mean_std(stats_df: pd.DataFrame, out_png: Path) -> None:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    lc_order = list(LC_GROUPS.keys())
    plt.figure(figsize=(13, 7), dpi=180)
    for i, season in enumerate(seasons):
        sub = stats_df[stats_df["Season"] == season].set_index("LC_Class").reindex(lc_order)
        y = sub["Mean_NSSR"].to_numpy(dtype=np.float32)
        yerr = sub["StdDev_NSSR"].to_numpy(dtype=np.float32)
        plt.errorbar(
            np.arange(len(lc_order)),
            y + (i - 1.5) * 0.6,
            xerr=yerr,
            fmt="o",
            capsize=3,
            label=season,
            alpha=0.85,
        )
    plt.yticks(np.arange(len(lc_order)), lc_order)
    plt.xlabel("NSSR mean ± std (W/m²)")
    plt.title("Seasonal NSSR Distribution Range by Land Cover")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _plot_normalized_means(stats_df: pd.DataFrame, out_png: Path) -> None:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    lc_order = list(LC_GROUPS.keys())
    pivot = stats_df.pivot(index="LC_Class", columns="Season", values="Mean_NSSR").reindex(lc_order)

    norm = pivot.copy()
    for season in seasons:
        col = norm[season]
        smean = np.nanmean(col.to_numpy(dtype=np.float64))
        if np.isfinite(smean) and smean > 0:
            norm[season] = col / smean
        else:
            norm[season] = np.nan

    x = np.arange(len(lc_order), dtype=np.float32)
    width = 0.18
    plt.figure(figsize=(13, 6), dpi=180)
    for i, season in enumerate(seasons):
        vals = norm[season].to_numpy(dtype=np.float32)
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=season)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.xticks(x, lc_order, rotation=25)
    plt.ylabel("Normalized Mean NSSR (class / season mean)")
    plt.title("Season-normalized NSSR by Land Cover Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _plot_spatial_maps(out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=180)
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    # Use a global color scale across seasons to avoid visual jump.
    all_vals = []
    season_arrays = {}
    for season in season_order:
        with rasterio.open(SEASON_FILES[season]) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None and np.isfinite(nodata):
                arr[arr == nodata] = np.nan
        season_arrays[season] = arr
        vals = arr[np.isfinite(arr) & (arr > 0)]
        if vals.size > 0:
            all_vals.append(vals)
    if not all_vals:
        raise ValueError("No valid NSSR values for spatial maps.")
    all_cat = np.concatenate(all_vals)
    vmin = float(np.nanpercentile(all_cat, 2))
    vmax = float(np.nanpercentile(all_cat, 98))

    for i, season in enumerate(season_order):
        arr = season_arrays[season]
        ax = axes.flat[i]
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(season)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    with rasterio.open(LC_PATH) as src_lc:
        lc = src_lc.read(1).astype(np.int16)
    ax_lc = axes.flat[4]
    ax_lc.imshow(lc, cmap="tab20")
    ax_lc.set_title("Land Cover (ESA LC 2023)")
    ax_lc.axis("off")

    axes.flat[5].axis("off")
    axes.flat[5].text(
        0.02,
        0.98,
        "2D analysis focus:\n"
        "1) Class-wise seasonal means\n"
        "2) Seasonal distribution range\n"
        "3) Spatial heterogeneity maps",
        va="top",
        ha="left",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _plot_seasonal_mean_share(stats_df: pd.DataFrame, share_df: pd.DataFrame, out_png: Path) -> None:
    """Four seasonal panels: 8-class mean NSSR (bar) + share (line)."""
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    lc_order = list(LC_GROUPS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=180)

    for ax, season in zip(axes.flat, seasons):
        s1 = stats_df[stats_df["Season"] == season].set_index("LC_Class").reindex(lc_order)
        s2 = share_df[share_df["Season"] == season].set_index("LC_Class").reindex(lc_order)

        x = np.arange(len(lc_order))
        mean_vals = s1["Mean_NSSR"].to_numpy(dtype=np.float32)
        share_vals = (s2["NSSR_Share"].to_numpy(dtype=np.float32) * 100.0)

        bars = ax.bar(x, mean_vals, color="#4C78A8", alpha=0.85, label="Mean NSSR")
        ax.set_title(season)
        ax.set_ylabel("Mean NSSR (W/m²)")
        ax.set_xticks(x)
        ax.set_xticklabels(lc_order, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(x, share_vals, color="#F58518", marker="o", linewidth=1.6, label="Contribution Share")
        ax2.set_ylabel("Contribution Share (%)")
        ax2.set_ylim(0, max(65, np.nanmax(share_vals) * 1.2 if np.isfinite(np.nanmax(share_vals)) else 65))

        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.suptitle("Seasonal NSSR by 8 Land-Cover Classes: Mean and Contribution Share", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()


def _plot_quantile_spread(stats_df: pd.DataFrame, out_png: Path) -> None:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    lc_order = list(LC_GROUPS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=180)
    for ax, season in zip(axes.flat, seasons):
        sub = stats_df[stats_df["Season"] == season].set_index("LC_Class").reindex(lc_order)
        x = np.arange(len(lc_order))
        p10 = sub["P10_NSSR"].to_numpy(dtype=np.float32)
        p50 = sub["P50_NSSR"].to_numpy(dtype=np.float32)
        p90 = sub["P90_NSSR"].to_numpy(dtype=np.float32)
        ax.bar(x, p90 - p10, bottom=p10, color="#93c5fd", alpha=0.7, label="P10-P90 range")
        ax.plot(x, p50, color="#1d4ed8", marker="o", linewidth=1.8, label="Median (P50)")
        ax.set_title(season)
        ax.set_xticks(x)
        ax.set_xticklabels(lc_order, rotation=25, ha="right")
        ax.set_ylabel("NSSR (W/m²)")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Seasonal NSSR Distribution by Land-Cover Class (P10-P90 + Median)", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()


def _write_summary(stats_df: pd.DataFrame, delta_df: pd.DataFrame, out_txt: Path) -> None:
    lines: list[str] = []
    lines.append("2D Seasonal NSSR Quantitative Assessment Summary")
    lines.append("")
    lines.append("Class-level mean NSSR by season:")
    lines.append(stats_df.to_string(index=False))
    lines.append("")
    lines.append("Buildings vs Natural (Grassland+Forest) delta:")
    lines.append(delta_df.to_string(index=False))
    lines.append("")
    bstd = stats_df[stats_df["LC_Class"] == "Buildings"][["Season", "StdDev_NSSR"]].set_index("Season")
    if "Summer" in bstd.index and "Winter" in bstd.index and bstd.loc["Winter", "StdDev_NSSR"] > 0:
        ratio = float(bstd.loc["Summer", "StdDev_NSSR"] / bstd.loc["Winter", "StdDev_NSSR"])
        lines.append(f"Buildings summer/winter std ratio: {ratio:.3f}")
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for p in SEASON_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing NSSR files:\n" + "\n".join(missing))
    if not LC_PATH.exists():
        raise FileNotFoundError(f"Missing land cover raster: {LC_PATH}")

    frames = []
    for season, nssr_path in SEASON_FILES.items():
        frames.append(_season_stats(season, nssr_path))

    stats_df = pd.concat(frames, ignore_index=True)
    delta_df = _build_delta_table(stats_df)

    stats_csv = OUT_DIR / "NSSR_LC_Seasonal_Statistics.csv"
    delta_csv = OUT_DIR / "NSSR_Delta_Buildings_vs_Natural.csv"
    share_csv = OUT_DIR / "NSSR_LC_Seasonal_Contribution.csv"
    core_suburb_csv = OUT_DIR / "NSSR_Core_Suburb_Seasonal_Statistics.csv"
    change_csv = OUT_DIR / "NSSR_LC_Seasonal_ChangeSummary.csv"
    fig_norm_png = FIG_DIR / "NSSR_LC_Seasonal_Mean_Normalized.png"
    fig_maps_png = FIG_DIR / "NSSR_Seasonal_Spatial_Maps.png"
    fig_mean_share_png = FIG_DIR / "NSSR_LC_Seasonal_Mean_Contribution_4Panel.png"
    fig_quantile_png = FIG_DIR / "NSSR_LC_Seasonal_QuantileSpread_4Panel.png"
    summary_txt = OUT_DIR / "NSSR_2D_Summary.txt"
    gradient_csv = OUT_DIR / "NSSR_Core_Suburb_Gradient.csv"

    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    delta_df.to_csv(delta_csv, index=False, encoding="utf-8-sig")
    share_df = _build_contribution_table(stats_df)
    share_df.to_csv(share_csv, index=False, encoding="utf-8-sig")
    core_suburb_frames = []
    for season, nssr_path in SEASON_FILES.items():
        core_suburb_frames.append(_core_suburb_stats(season, nssr_path))
    core_suburb_df = pd.concat(core_suburb_frames, ignore_index=True)
    core_suburb_df.to_csv(core_suburb_csv, index=False, encoding="utf-8-sig")
    gradient_df = _build_core_suburb_gradient(core_suburb_df)
    gradient_df.to_csv(gradient_csv, index=False, encoding="utf-8-sig")
    change_df = _season_change_summary(stats_df)
    change_df.to_csv(change_csv, index=False, encoding="utf-8-sig")
    _plot_normalized_means(stats_df, fig_norm_png)
    _plot_spatial_maps(fig_maps_png)
    _plot_seasonal_mean_share(stats_df, share_df, fig_mean_share_png)
    _plot_quantile_spread(stats_df, fig_quantile_png)
    _write_summary(stats_df, delta_df, summary_txt)

    print(f"Saved: {stats_csv}")
    print(f"Saved: {delta_csv}")
    print(f"Saved: {share_csv}")
    print(f"Saved: {core_suburb_csv}")
    print(f"Saved: {gradient_csv}")
    print(f"Saved: {change_csv}")
    print(f"Saved: {fig_norm_png}")
    print(f"Saved: {fig_maps_png}")
    print(f"Saved: {fig_mean_share_png}")
    print(f"Saved: {fig_quantile_png}")
    print(f"Saved: {summary_txt}")
    print("\nSeasonal LC statistics preview:")
    print(stats_df.to_string(index=False))
    print("\nDelta NSSR preview:")
    print(delta_df.to_string(index=False))
    print("\nContribution preview:")
    print(share_df.to_string(index=False))
    print("\nCore/Suburb preview:")
    print(core_suburb_df.head(20).to_string(index=False))
    print("\nSeasonal change summary preview:")
    print(change_df.to_string(index=False))


if __name__ == "__main__":
    main()
