"""
generate_demo_models.py
=======================
Generates synthetic but realistic demo artefacts for the Streamlit app.

Run this script once to create ``saved_models/`` so the app can launch
without requiring access to the original Rimba Raya training data or
Google Earth Engine.

    python generate_demo_models.py

Author : <Your Name>
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

MODEL_FEATURES = [
    "clay", "sand", "silt", "elev", "annual_precip",
    "peat_frac", "Hydraulic_Stress", "NDVI",
    "precip_clay_interaction",
]
BASE_MODEL_NAMES = ["RF", "XGB", "SVR"]
N_SAMPLES = 500
RANDOM_STATE = 42
OUTPUT_DIR = "saved_models"


def _generate_synthetic_dataset(n: int = N_SAMPLES) -> tuple:
    """
    Produce a realistic synthetic dataset that mimics the covariate
    distributions observed in the Rimba Raya tropical peat-swamp forest.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    clay = rng.uniform(15, 60, n)
    sand = rng.uniform(5, 40, n)
    silt = 100 - clay - sand
    elev = rng.uniform(0, 30, n)
    annual_precip = rng.uniform(2000, 4500, n)
    peat_frac = rng.uniform(0.3, 1.0, n)
    Hydraulic_Stress = rng.uniform(0.05, 0.8, n)
    NDVI = rng.uniform(0.5, 0.95, n)
    precip_clay_interaction = annual_precip * clay

    X = pd.DataFrame({
        "clay": clay, "sand": sand, "silt": silt, "elev": elev,
        "annual_precip": annual_precip, "peat_frac": peat_frac,
        "Hydraulic_Stress": Hydraulic_Stress, "NDVI": NDVI,
        "precip_clay_interaction": precip_clay_interaction,
    })

    # Plausible AGB response for tropical peat forest (50–400 Mg/ha)
    y = (
        80
        + 0.5 * clay
        + 0.03 * annual_precip
        + 120 * peat_frac
        + 100 * NDVI
        - 50 * Hydraulic_Stress
        - 0.2 * elev
        + 0.0001 * precip_clay_interaction
        + rng.normal(0, 20, n)
    )
    y = np.clip(y, 30, 450)

    return X, y


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = _generate_synthetic_dataset()

    # Preprocessing
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(imputer.fit_transform(X))

    # Base models
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)
    xgb = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        objective="reg:squarederror", verbosity=0, random_state=RANDOM_STATE,
    )
    svr = SVR(C=10, kernel="rbf")

    rf.fit(X_sc, y)
    xgb.fit(X_sc, y)
    svr.fit(X_sc, y)

    # Generate meta-features (OOF approximated here for demo)
    meta_X = pd.DataFrame({
        "RF": rf.predict(X_sc),
        "XGB": xgb.predict(X_sc),
        "SVR": svr.predict(X_sc),
    })

    # Meta-model
    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_X, y)

    # Persist artefacts
    joblib.dump(ridge, os.path.join(OUTPUT_DIR, "meta_model.joblib"))
    joblib.dump(imputer, os.path.join(OUTPUT_DIR, "imputer.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "base_model_RF.joblib"))
    joblib.dump(xgb, os.path.join(OUTPUT_DIR, "base_model_XGB.joblib"))
    joblib.dump(svr, os.path.join(OUTPUT_DIR, "base_model_SVR.joblib"))

    metadata = {
        "version": "1.0-demo",
        "created_at": datetime.utcnow().isoformat(),
        "features": MODEL_FEATURES,
        "base_models": BASE_MODEL_NAMES,
        "target": "AGB",
        "note": "Demo artefacts — replace with production models from stacking_model.py",
        "ridge_coefficients": dict(zip(BASE_MODEL_NAMES, ridge.coef_.tolist())),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"✅  Demo artefacts written to '{OUTPUT_DIR}/'")
    print(f"    Ridge weights: { {k: round(v, 4) for k, v in metadata['ridge_coefficients'].items()} }")


if __name__ == "__main__":
    main()
