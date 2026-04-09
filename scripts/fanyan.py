from __future__ import annotations

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import mapping


ROOT = Path(r"F:\project2025\wulifanyan")
XGB_INPUT = ROOT / "XGBoost" / "input"
OUTPUT_DIR = ROOT / "output" / "NSSR"
DIAG_DIR = ROOT / "output" / "NSSR_diag"
ROI_SHP = ROOT / "研究区" / "方形研究区.shp"

# NSSR shortwave-only parameterization constants
SOLAR_CONSTANT = 1367.0
LV_OVER_RV = 5423.0
ALBEDO_SCALE = 1.0
ALBEDO_MIN = 0.02
ALBEDO_MAX = 0.95

SEASON_CONFIG = {
    "Winter": {
        "date": "20230116",
        "beta": 0.08,
        "albedo": XGB_INPUT / "Albedo" / "Project_Final_Winter_20230116.tif",
        "era5": ROOT / "input" / "era52" / "ERA5_Pixel_Solar_Fix_Optimized_2023-01-16.tif",
    },
    "Spring": {
        "date": "20230305",
        "beta": 0.10,
        "albedo": XGB_INPUT / "Albedo" / "Project_Final_Spring_20230305.tif",
        "era5": ROOT / "input" / "era52" / "ERA5_Pixel_Solar_Fix_Optimized_2023-03-05.tif",
    },
    "Summer": {
        "date": "20230828",
        "beta": 0.12,
        "albedo": XGB_INPUT / "Albedo" / "Project_Final_Summer_20230828.tif",
        "era5": ROOT / "input" / "era52" / "ERA5_Pixel_Solar_Fix_Optimized_2023-08-28.tif",
    },
    "Autumn": {
        "date": "20231124",
        "beta": 0.09,
        "albedo": XGB_INPUT / "Albedo" / "Project_Final_Autumn_20231124.tif",
        "era5": ROOT / "input" / "era52" / "ERA5_Pixel_Solar_Fix_Optimized_2023-11-24.tif",
    },
}


def _pick_band_by_name(data: xr.DataArray, preferred_name: str, fallback_index: int) -> xr.DataArray:
    if "band" not in data.dims:
        return data
    descriptions = list(data.attrs.get("long_name", []))
    for idx, desc in enumerate(descriptions, start=1):
        if str(desc).lower() == preferred_name.lower():
            return data.sel(band=idx)
    if data.sizes["band"] >= fallback_index:
        return data.sel(band=fallback_index)
    return data.isel(band=0)


def _prepare_albedo(albedo_raw: xr.DataArray) -> xr.DataArray:
    albedo = _pick_band_by_name(albedo_raw, "Albedo", 7).astype("float32")
    # Guard: report suspicious scaling before clipping.
    amax = float(np.nanmax(albedo.values))
    if amax > 1.2 and ALBEDO_SCALE == 1.0:
        warnings.warn(
            f"Albedo max={amax:.3f} (>1). Consider setting ALBEDO_SCALE=0.0001 or 0.01 according to export scale."
        )
    albedo = albedo * np.float32(ALBEDO_SCALE)
    return albedo.clip(min=np.float32(ALBEDO_MIN), max=np.float32(ALBEDO_MAX))


def _prepare_era5(era5_raw: xr.DataArray, ref: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    td_c = _pick_band_by_name(era5_raw, "Td_Celsius", 1).astype("float32")
    dssr = _pick_band_by_name(era5_raw, "DSSR_Wm2", 4).astype("float32")
    zenith = _pick_band_by_name(era5_raw, "Solar_Zenith", 5).astype("float32")
    td_c = td_c.rio.reproject_match(ref, resampling=Resampling.bilinear)
    dssr = dssr.rio.reproject_match(ref, resampling=Resampling.bilinear)
    zenith = zenith.rio.reproject_match(ref, resampling=Resampling.bilinear)
    return td_c, dssr, zenith


def _calc_rs_down(td_c: xr.DataArray, zenith_deg: xr.DataArray, beta: float) -> xr.DataArray:
    td_k = td_c + 273.15
    cos_theta = np.cos(np.deg2rad(zenith_deg))
    cos_theta = xr.where(cos_theta > 0.0, cos_theta, np.nan)

    # e0 in hPa from dewpoint
    e0_hpa = 6.11 * np.exp(LV_OVER_RV * ((1.0 / 273.15) - (1.0 / td_k)))
    denominator = 1.085 * cos_theta + e0_hpa * (2.7 + cos_theta) * 1.0e-3 + np.float32(beta)
    rs_down = SOLAR_CONSTANT * (cos_theta ** 2) / denominator
    return rs_down.astype("float32")


def _save_diag_raster(data: xr.DataArray, geometries, crs, out_path: Path) -> None:
    clipped = data.rio.clip(geometries, crs, drop=True).astype("float32")
    clipped.attrs = {}
    clipped.rio.to_raster(out_path, compress="LZW")


def calculate_nssr_inversion(
    albedo_path: Path,
    era5_path: Path,
    shp_path: Path,
    output_path: Path,
    beta: float,
) -> None:
    roi = gpd.read_file(shp_path)
    geometries = [mapping(geom) for geom in roi.geometry if geom is not None]

    albedo_raw = rioxarray.open_rasterio(albedo_path, masked=True)
    albedo = _prepare_albedo(albedo_raw)
    albedo_mask = xr.where(
        (albedo <= np.float32(ALBEDO_MIN)) | (albedo >= np.float32(ALBEDO_MAX)),
        np.float32(1.0),
        np.float32(0.0),
    ).astype("float32")
    albedo_mask = albedo_mask.rio.write_crs(albedo.rio.crs)

    era5_raw = rioxarray.open_rasterio(era5_path, masked=True)
    td_c, dssr, zenith = _prepare_era5(era5_raw, albedo)

    # NSSR (shortwave net radiation) uses downward shortwave directly:
    # NSSR = (1 - Albedo) * DSSR
    rs_down = dssr.astype("float32")
    one_minus_albedo = (1.0 - albedo).astype("float32")
    nssr = (one_minus_albedo * rs_down).astype("float32")

    invalid = (~np.isfinite(albedo)) | (~np.isfinite(td_c)) | (~np.isfinite(zenith)) | (~np.isfinite(dssr))
    nssr = nssr.where(~invalid)
    nssr.name = "NSSR"

    # Diagnostic: parameterized Rs vs ERA5 DSSR
    bias_check = xr.where(np.isfinite(dssr) & (dssr > 0), rs_down / dssr, np.nan).astype("float32")

    nssr_clipped = nssr.rio.clip(geometries, roi.crs, drop=True)
    nssr_clipped = nssr_clipped.where((nssr_clipped >= 0.0) & (nssr_clipped <= 1367.0), np.nan)
    nssr_clipped = nssr_clipped.rio.write_nodata(np.nan)

    stem = output_path.stem
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    _save_diag_raster(rs_down, geometries, roi.crs, DIAG_DIR / f"{stem}_RsDown.tif")
    _save_diag_raster(dssr, geometries, roi.crs, DIAG_DIR / f"{stem}_DSSR.tif")
    _save_diag_raster(bias_check, geometries, roi.crs, DIAG_DIR / f"{stem}_RsDownOverDSSR.tif")
    _save_diag_raster(albedo, geometries, roi.crs, DIAG_DIR / f"{stem}_AlbedoUsed.tif")
    _save_diag_raster(albedo_mask, geometries, roi.crs, DIAG_DIR / f"{stem}_AlbedoMask.tif")
    _save_diag_raster(one_minus_albedo, geometries, roi.crs, DIAG_DIR / f"{stem}_OneMinusAlbedo.tif")
    _save_diag_raster(nssr, geometries, roi.crs, DIAG_DIR / f"{stem}_NSSRRaw.tif")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nssr_clipped.rio.to_raster(output_path, compress="LZW")
    ratio_vals = bias_check.values
    finite_ratio = ratio_vals[np.isfinite(ratio_vals)]
    ratio_mean = float(np.nanmean(finite_ratio)) if finite_ratio.size else np.nan
    ratio_min = float(np.nanmin(finite_ratio)) if finite_ratio.size else np.nan
    ratio_max = float(np.nanmax(finite_ratio)) if finite_ratio.size else np.nan
    print(
        f"Saved: {output_path} | RsDown/DSSR mean={ratio_mean:.4f}, min={ratio_min:.4f}, max={ratio_max:.4f}"
    )
    return {
        "output": str(output_path),
        "ratio_mean": ratio_mean,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
    }


def main() -> None:
    if not ROI_SHP.exists():
        raise FileNotFoundError(f"Missing ROI shapefile: {ROI_SHP}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        cfg = SEASON_CONFIG[season]
        out_path = OUTPUT_DIR / f"NSSR_Final_Inversion_{season}_{cfg['date']}.tif"
        info = calculate_nssr_inversion(
            cfg["albedo"],
            cfg["era5"],
            ROI_SHP,
            out_path,
            beta=float(cfg["beta"]),
        )
        info["season"] = season
        summary.append(info)

    if summary:
        import csv

        csv_path = DIAG_DIR / "NSSR_validation_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=["season", "output", "ratio_mean", "ratio_min", "ratio_max"]
            )
            writer.writeheader()
            writer.writerows(summary)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
