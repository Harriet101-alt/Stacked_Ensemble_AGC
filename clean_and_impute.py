"""
clean_and_impute.py
===================
Data cleaning pipeline for the Rimba Raya biomass dataset.

Transforms the raw CSV (rimba_raya_biomass_with_all_features_raw.csv) into a
model-ready dataset (rimba_raya_final.csv) by:

1. Parsing dict-string columns to extract median values
2. Fixing column mappings (annual_precip, peat_frac)
3. Recomputing precip_clay_interaction with corrected precipitation
4. Spatially imputing missing groundwater_depth values

Author : Harriet Fletcher
"""

from __future__ import annotations

import ast
import math
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(SCRIPT_DIR, "rimba_raya_biomass_with_all_features_raw.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "rimba_raya_final.csv")

# Columns expected by stacking_model.py
MODEL_FEATURES = [
    "clay", "sand", "silt", "elev",
    "annual_precip", "peat_frac",
    "Hydraulic_Stress", "NDVI",
    "precip_clay_interaction",
]

OUTPUT_COLUMNS = [
    "lat", "lon", "year", "AGB",
    *MODEL_FEATURES,
    "groundwater_depth",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_dict_column(series: pd.Series) -> pd.Series:
    """Extract the 'median' value from stringified Python dicts."""
    def _extract(val):
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                d = ast.literal_eval(val)
                return d.get("median")
            except (ValueError, SyntaxError):
                return np.nan
        return val
    return series.apply(_extract)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Step 1: Load raw data ─────────────────────────────────────────────
    print(f"Loading raw data from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Step 1b: Parse dict-string columns ────────────────────────────────
    dict_cols = [
        col for col in df.columns
        if df[col].dtype == object
        and df[col].dropna().astype(str).str.startswith("{").any()
    ]
    print(f"  Parsing {len(dict_cols)} dict-string columns: {dict_cols[:5]}...")
    for col in dict_cols:
        df[col] = parse_dict_column(df[col])

    # ── Step 2: Fix column mappings ───────────────────────────────────────
    # annual_precip currently contains year values; actual precipitation
    # is in total_precipitation
    print("\nFixing column mappings:")

    if "total_precipitation" in df.columns:
        df["annual_precip"] = df["total_precipitation"]
        print(f"  annual_precip ← total_precipitation "
              f"(range: {df['annual_precip'].min():.1f} – "
              f"{df['annual_precip'].max():.1f})")
    else:
        raise KeyError("Expected column 'total_precipitation' not found in raw data.")

    if "peatland_fraction" in df.columns:
        df["peat_frac"] = df["peatland_fraction"]
        print(f"  peat_frac ← peatland_fraction "
              f"(range: {df['peat_frac'].min():.4f} – "
              f"{df['peat_frac'].max():.4f})")
    elif "peat" in df.columns:
        df["peat_frac"] = df["peat"]
        print(f"  peat_frac ← peat")
    else:
        raise KeyError("Neither 'peatland_fraction' nor 'peat' found in raw data.")

    # Recompute interaction term with corrected precipitation
    df["precip_clay_interaction"] = df["annual_precip"] * df["clay"]
    print(f"  precip_clay_interaction recomputed "
          f"(range: {df['precip_clay_interaction'].min():.1f} – "
          f"{df['precip_clay_interaction'].max():.1f})")

    # ── Step 3: Spatial imputation of groundwater_depth ───────────────────
    print("\nImputing groundwater_depth:")
    n_missing = df["groundwater_depth"].isna().sum()
    print(f"  Missing values: {n_missing}/{len(df)}")

    if n_missing > 0:
        coords = df.groupby(["lat", "lon"])
        coord_means = coords["groundwater_depth"].transform("mean")

        # Primary: fill with same-coordinate temporal mean
        filled = df["groundwater_depth"].fillna(coord_means)
        remaining = filled.isna().sum()

        if remaining > 0:
            # Fallback: nearest coordinate with data
            print(f"  {remaining} values still missing after temporal mean — "
                  f"using nearest-coordinate fallback")
            coord_mean_lookup = (
                df.groupby(["lat", "lon"])["groundwater_depth"]
                .mean()
                .dropna()
            )
            for idx in filled[filled.isna()].index:
                row_lat, row_lon = df.loc[idx, "lat"], df.loc[idx, "lon"]
                best_dist = float("inf")
                best_val = np.nan
                for (clat, clon), mean_val in coord_mean_lookup.items():
                    d = haversine_km(row_lat, row_lon, clat, clon)
                    if d < best_dist:
                        best_dist = d
                        best_val = mean_val
                filled.iloc[idx] = best_val

        df["groundwater_depth"] = filled
        print(f"  After imputation: {df['groundwater_depth'].isna().sum()} missing")

    # ── Step 3b: Impute NDVI (same-coordinate temporal mean) ──────────────
    n_ndvi_missing = df["NDVI"].isna().sum()
    if n_ndvi_missing > 0:
        print(f"\nImputing NDVI:")
        print(f"  Missing values: {n_ndvi_missing}/{len(df)}")
        ndvi_coord_means = df.groupby(["lat", "lon"])["NDVI"].transform("mean")
        df["NDVI"] = df["NDVI"].fillna(ndvi_coord_means)
        print(f"  After imputation: {df['NDVI'].isna().sum()} missing")

    # ── Step 4: Assemble final DataFrame and save ─────────────────────────
    df_final = df[OUTPUT_COLUMNS].copy()

    # Validation
    nan_counts = df_final.isna().sum()
    has_nan = nan_counts.any()
    if has_nan:
        print("\nWARNING — NaN values remain:")
        print(nan_counts[nan_counts > 0])
    else:
        print("\nValidation passed: no NaN in any output column.")

    df_final.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {OUT_PATH}")
    print(f"  Shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")

    # Summary statistics
    print("\n── Summary Statistics ──")
    print(df_final.describe().round(4).to_string())


if __name__ == "__main__":
    main()
