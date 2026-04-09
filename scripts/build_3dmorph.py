from __future__ import annotations

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box


ROOT = Path(r"F:\project2025\wulifanyan")
BUILDING_SHP = ROOT / "input" / "建筑物轮廓数据" / "北京.shp"
ROI_SHP = ROOT / "研究区" / "方形研究区.shp"
SVF_TIF = ROOT / "XGBoost" / "input" / "svfdata" / "svf200_30mv2.tif"
OUT_DIR = ROOT / "output" / "3dmorph"
OUT_TIF = OUT_DIR / "Morphology_30m_with_SVF.tif"
OUT_CSV = OUT_DIR / "Morphology_30m_metrics_table.csv"
OUT_GPKG = OUT_DIR / "Morphology_30m_metrics_grid.gpkg"

TARGET_CRS = "EPSG:32650"
GRID_AREA = 900.0  # 30m x 30m


def _make_fishnet_from_template(roi: gpd.GeoDataFrame, template_path: Path) -> tuple[gpd.GeoDataFrame, dict]:
    with rasterio.open(template_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        crs = src.crs
        svf = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()

    roi_proj = roi.to_crs(crs)
    mask = geometry_mask(
        [geom for geom in roi_proj.geometry if geom is not None],
        transform=transform,
        invert=True,
        out_shape=(height, width),
    )

    rows, cols = np.where(mask)
    a = transform.a
    e = transform.e
    c0 = transform.c
    f0 = transform.f

    geoms = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        x_left = c0 + c * a
        y_top = f0 + r * e
        geoms.append(box(x_left, y_top + e, x_left + a, y_top))

    fishnet = gpd.GeoDataFrame(
        {
            "grid_id": np.arange(len(rows), dtype=np.int64),
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
        },
        geometry=geoms,
        crs=crs,
    )

    svf = np.where(mask, svf, np.nan).astype(np.float32)
    if nodata is not None:
        svf = np.where(np.isclose(svf, nodata), np.nan, svf).astype(np.float32)

    return fishnet, {"svf": svf, "profile": profile, "mask": mask}


def _prepare_buildings(building_path: Path, roi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bldg = gpd.read_file(building_path)
    if bldg.crs is None:
        raise ValueError("Building SHP has no CRS.")
    bldg = bldg.to_crs(TARGET_CRS)
    roi_proj = roi.to_crs(TARGET_CRS)

    required = {"H", "Floor"}
    missing = required.difference(bldg.columns)
    if missing:
        raise ValueError(f"Missing required fields in building SHP: {sorted(missing)}")

    bldg = bldg[["H", "Floor", "geometry"]].copy()
    bldg["H"] = pd.to_numeric(bldg["H"], errors="coerce").astype(np.float64)
    bldg["Floor"] = pd.to_numeric(bldg["Floor"], errors="coerce").astype(np.float64)
    bldg = bldg[np.isfinite(bldg["H"]) & (bldg["H"] > 0)].copy()

    # Geometry safety.
    bldg.geometry = bldg.geometry.make_valid()
    bldg.geometry = bldg.buffer(0)
    bldg = bldg[bldg.geometry.notna() & (~bldg.geometry.is_empty)].copy()

    # Clip by ROI before intersection.
    bldg_clip = gpd.overlay(bldg, roi_proj[["geometry"]], how="intersection", keep_geom_type=False)
    bldg_clip = bldg_clip[bldg_clip.geometry.notna() & (~bldg_clip.geometry.is_empty)].copy()
    return bldg_clip


def _compute_metrics(fishnet: gpd.GeoDataFrame, bldg_clip: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if bldg_clip.empty:
        out = fishnet.copy()
        for field in ["MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI"]:
            out[field] = 0.0
        return out

    inter = gpd.overlay(
        fishnet[["grid_id", "row", "col", "geometry"]],
        bldg_clip[["H", "Floor", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    inter = inter[inter.geometry.notna() & (~inter.geometry.is_empty)].copy()
    inter["footprint"] = inter.geometry.area.astype(np.float64)
    inter["perimeter"] = inter.geometry.length.astype(np.float64)
    inter = inter[inter["footprint"] > 0].copy()

    if inter.empty:
        out = fishnet.copy()
        for field in ["MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI"]:
            out[field] = 0.0
        return out

    inter["Floor_use"] = np.where(
        np.isfinite(inter["Floor"]) & (inter["Floor"] > 0),
        inter["Floor"],
        inter["H"] / 3.0,
    )

    inter["vol_part"] = inter["footprint"] * inter["H"]
    inter["far_part"] = inter["footprint"] * inter["Floor_use"]
    inter["bsa_part"] = inter["perimeter"] * inter["H"] + inter["footprint"]
    inter["bsi_part"] = np.where(inter["H"] > 0, inter["footprint"] / inter["H"], np.nan)

    grouped = inter.groupby("grid_id", as_index=False).agg(
        MBH=("H", "mean"),
        HMAX=("H", "max"),
        HMIN=("H", "min"),
        FAR_NUM=("far_part", "sum"),
        MBV=("vol_part", "mean"),
        VOL_SUM=("vol_part", "sum"),
        BSA=("bsa_part", "sum"),
        BSI=("bsi_part", "mean"),
    )

    grouped["MBH"] = grouped["MBH"].round(1)
    grouped["BU"] = grouped["HMAX"] - grouped["HMIN"]
    grouped["FAR"] = grouped["FAR_NUM"] / GRID_AREA
    grouped["SCD"] = np.where(grouped["HMAX"] > 0, grouped["VOL_SUM"] / (grouped["HMAX"] * GRID_AREA), 0.0)

    keep = grouped[["grid_id", "MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI"]].copy()
    out = fishnet.merge(keep, on="grid_id", how="left")
    for c in ["MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI"]:
        out[c] = out[c].fillna(0.0).astype(np.float32)
    return out


def _write_multiband(metrics: gpd.GeoDataFrame, template_meta: dict, out_tif: Path) -> None:
    svf = template_meta["svf"].astype(np.float32)
    profile = template_meta["profile"].copy()
    height, width = svf.shape

    bands = {
        "MBH": np.full((height, width), np.nan, dtype=np.float32),
        "BU": np.full((height, width), np.nan, dtype=np.float32),
        "FAR": np.full((height, width), np.nan, dtype=np.float32),
        "MBV": np.full((height, width), np.nan, dtype=np.float32),
        "SCD": np.full((height, width), np.nan, dtype=np.float32),
        "BSA": np.full((height, width), np.nan, dtype=np.float32),
        "BSI": np.full((height, width), np.nan, dtype=np.float32),
    }

    rows = metrics["row"].to_numpy(dtype=np.int64)
    cols = metrics["col"].to_numpy(dtype=np.int64)
    for name in bands:
        bands[name][rows, cols] = metrics[name].to_numpy(dtype=np.float32)

    out_stack = np.stack(
        [
            bands["MBH"],
            bands["BU"],
            bands["FAR"],
            bands["MBV"],
            bands["SCD"],
            bands["BSA"],
            bands["BSI"],
            svf,
        ],
        axis=0,
    )

    profile.update(
        driver="GTiff",
        count=8,
        dtype="float32",
        nodata=np.nan,
        compress="LZW",
    )

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out_stack)
        dst.set_band_description(1, "MBH")
        dst.set_band_description(2, "BU")
        dst.set_band_description(3, "FAR")
        dst.set_band_description(4, "MBV")
        dst.set_band_description(5, "SCD")
        dst.set_band_description(6, "BSA")
        dst.set_band_description(7, "BSI")
        dst.set_band_description(8, "SVF")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roi = gpd.read_file(ROI_SHP)
    if roi.crs is None:
        raise ValueError("ROI SHP has no CRS.")

    fishnet, template_meta = _make_fishnet_from_template(roi, SVF_TIF)
    print(f"Fishnet cells in ROI: {len(fishnet):,}")

    bldg_clip = _prepare_buildings(BUILDING_SHP, roi)
    print(f"Buildings after ROI clip: {len(bldg_clip):,}")

    metrics = _compute_metrics(fishnet, bldg_clip)
    metrics["SVF"] = template_meta["svf"][metrics["row"], metrics["col"]].astype(np.float32)

    _write_multiband(metrics, template_meta, OUT_TIF)
    metrics.drop(columns="geometry").to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    metrics.to_file(OUT_GPKG, layer="grid_metrics", driver="GPKG")

    print(f"Saved: {OUT_TIF}")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_GPKG}")


if __name__ == "__main__":
    main()

