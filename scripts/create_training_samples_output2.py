from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray


ROOT = Path(r"F:\project2025\wulifanyan")
XGB_ROOT = ROOT / "XGBoost"
OUT_DIR = XGB_ROOT / "output2"
TRAIN_DIR = OUT_DIR / "train"

MORPH_PATH = ROOT / "output" / "3dmorph" / "Morphology_30m_with_SVF.tif"
NSSR_PATHS = {
    "Spring": ROOT / "output" / "NSSR" / "NSSR_Final_Inversion_Spring_20230305.tif",
    "Summer": ROOT / "output" / "NSSR" / "NSSR_Final_Inversion_Summer_20230828.tif",
    "Autumn": ROOT / "output" / "NSSR" / "NSSR_Final_Inversion_Autumn_20231124.tif",
    "Winter": ROOT / "output" / "NSSR" / "NSSR_Final_Inversion_Winter_20230116.tif",
}

FACTOR_BANDS = {
    "MBH": 1,
    "BU": 2,
    "FAR": 3,
    "MBV": 4,
    "SCD": 5,
    "BSA": 6,
    "BSI": 7,
    "SVF": 8,
}


def _extract_xy(nssr: rioxarray.DataArray) -> tuple[np.ndarray, np.ndarray]:
    x_coords = np.asarray(nssr.x.values, dtype=np.float64)
    y_coords = np.asarray(nssr.y.values, dtype=np.float64)
    xx, yy = np.meshgrid(x_coords, y_coords)
    return xx.astype(np.float32), yy.astype(np.float32)


def _season_df(season: str, morph: rioxarray.DataArray, nssr_path: Path) -> pd.DataFrame:
    nssr = rioxarray.open_rasterio(nssr_path, masked=True).squeeze()
    morph_aligned = morph.rio.reproject_match(nssr)

    data = {
        "Season": np.full(nssr.shape, season, dtype=object),
        "NSSR": np.asarray(nssr.values, dtype=np.float32),
    }
    for name, band in FACTOR_BANDS.items():
        data[name] = np.asarray(morph_aligned.sel(band=band).values, dtype=np.float32)

    xx, yy = _extract_xy(nssr)
    data["X"] = xx
    data["Y"] = yy

    flat = {k: v.reshape(-1) for k, v in data.items()}
    df = pd.DataFrame(flat)
    return df


def main() -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    morph = rioxarray.open_rasterio(MORPH_PATH, masked=True)

    all_df: list[pd.DataFrame] = []
    required = ["NSSR", "MBH", "BU", "FAR", "MBV", "SCD", "BSA", "BSI", "SVF", "X", "Y"]

    for season, path in NSSR_PATHS.items():
        df = _season_df(season, morph, path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
        # Decoupled seasonal training: keep building-influenced samples.
        df = df[(df["NSSR"] > 0) & (df["MBH"] > 0)].copy()

        season_csv = TRAIN_DIR / f"Training_Samples_{season}_Output2.csv"
        df.to_csv(season_csv, index=False, encoding="utf-8-sig")
        print(f"Saved: {season_csv} rows={len(df):,}")
        all_df.append(df)

    full_df = pd.concat(all_df, ignore_index=True)
    full_csv = TRAIN_DIR / "Training_Samples_Full_Output2.csv"
    full_df.to_csv(full_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {full_csv} rows={len(full_df):,}")


if __name__ == "__main__":
    main()

