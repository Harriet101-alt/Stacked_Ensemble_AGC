"""
inference_engine.py
===================
Stateless inference layer for the Biomass Stacked Ensemble.

This module decouples prediction logic from the Streamlit UI, making it
independently testable and reusable by a REST API or batch pipeline.

Classes
-------
InputPayload   — Pydantic model for input validation with domain bounds.
InferenceEngine — Two-stage prediction pipeline (8 covariates → 3 base
                  predictions → 1 Ridge meta-prediction).

Author : <Your Name>
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Input validation schema
# ---------------------------------------------------------------------------


class InputPayload(BaseModel):
    """
    Pydantic model enforcing domain-specific bounds for all eight raw
    environmental covariates collected from the Streamlit sidebar sliders.

    All bounds are derived from the Rimba Raya dataset statistics and
    physical plausibility constraints for a tropical peat-swamp forest.
    """

    clay: float = Field(
        ..., ge=0.0, le=100.0, description="Clay content of topsoil (%)"
    )
    sand: float = Field(
        ..., ge=0.0, le=100.0, description="Sand content of topsoil (%)"
    )
    silt: float = Field(
        ..., ge=0.0, le=100.0, description="Silt content of topsoil (%)"
    )
    elev: float = Field(
        ..., ge=0.0, le=500.0, description="Mean elevation above sea level (m)"
    )
    annual_precip: float = Field(
        ..., ge=500.0, le=6000.0, description="Annual precipitation (mm/yr)"
    )
    peat_frac: float = Field(
        ..., ge=0.0, le=1.0, description="Peatland fraction (0–1)"
    )
    Hydraulic_Stress: float = Field(
        ..., ge=0.0, le=2.0, description="Root-zone moisture / (precip + 1)"
    )
    NDVI: float = Field(
        ..., ge=-1.0, le=1.0, description="Normalised Difference Vegetation Index"
    )

    @field_validator("clay", "sand", "silt")
    @classmethod
    def texture_bounds(cls, v: float) -> float:
        """Soil texture fractions must lie within [0, 100]."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"Soil texture value {v} is outside [0, 100].")
        return v

    @field_validator("NDVI")
    @classmethod
    def ndvi_bounds(cls, v: float) -> float:
        """NDVI is bounded in [-1, 1] by definition."""
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"NDVI {v} is outside the physical range [-1, 1].")
        return v


# ---------------------------------------------------------------------------
# Feature engineering helper
# ---------------------------------------------------------------------------


def engineer_features(payload: InputPayload) -> Dict[str, float]:
    """
    Expand an ``InputPayload`` into the full 9-feature dict expected by
    the pre-fitted scaler (8 raw covariates + 1 derived interaction term).

    The interaction term ``precip_clay_interaction`` captures the
    amplifying effect of high precipitation in clay-rich soils on
    waterlogging stress, which is a known driver of above-ground biomass
    distribution in tropical peat-swamp forests.

    Parameters
    ----------
    payload : InputPayload

    Returns
    -------
    feature_dict : dict[str, float]
        Keys match ``MODEL_FEATURES`` in ``stacking_model.py`` exactly.
    """
    data = payload.model_dump()
    data["precip_clay_interaction"] = data["annual_precip"] * data["clay"]
    return data


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

# Canonical feature order — must match training column order exactly
MODEL_FEATURES: List[str] = [
    "clay",
    "sand",
    "silt",
    "elev",
    "annual_precip",
    "peat_frac",
    "Hydraulic_Stress",
    "NDVI",
    "precip_clay_interaction",
]

BASE_MODEL_NAMES: List[str] = ["RF", "XGB", "SVR"]


@dataclass
class PredictionResult:
    """Container for a complete inference result."""

    final_prediction: float
    base_predictions: Dict[str, float]
    meta_weights: Dict[str, float]
    interaction_value: float
    sensitivity_log: List[str] = field(default_factory=list)


class InferenceEngine:
    """
    Stateless, two-stage inference pipeline.

    Stage 1 — Transform 8 raw environmental covariates into 3 Level-0
               predictions (RF, XGB, SVR) using the persisted preprocessing
               transforms and base model weights.

    Stage 2 — Combine the 3 base predictions via the Ridge meta-model to
               produce a single biomass estimate (Mg/ha).

    Parameters
    ----------
    model_dir : str
        Path to the directory produced by
        ``StackingOrchestrator.save_artefacts()``.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._meta_model = None
        self._base_models: Dict[str, object] = {}
        self._imputer = None
        self._scaler = None
        self._metadata: Dict = {}
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "InferenceEngine":
        """
        Deserialise all artefacts from ``model_dir``.

        Returns *self* to allow method chaining:
        ``engine = InferenceEngine("saved_models").load()``
        """
        self._meta_model = joblib.load(
            os.path.join(self.model_dir, "meta_model.joblib")
        )
        self._imputer = joblib.load(
            os.path.join(self.model_dir, "imputer.joblib")
        )
        self._scaler = joblib.load(
            os.path.join(self.model_dir, "scaler.joblib")
        )

        for name in BASE_MODEL_NAMES:
            path = os.path.join(
                self.model_dir, f"base_model_{name}.joblib"
            )
            self._base_models[name] = joblib.load(path)

        metadata_path = os.path.join(self.model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as fh:
                self._metadata = json.load(fh)

        self._is_loaded = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, payload: InputPayload) -> PredictionResult:
        """
        Execute the full two-stage inference pipeline.

        Parameters
        ----------
        payload : InputPayload
            Validated slider values from the Streamlit UI.

        Returns
        -------
        PredictionResult
            Includes the final biomass estimate, individual base model
            predictions, Ridge meta-weights, and a human-readable
            sensitivity log.

        Raises
        ------
        RuntimeError
            If ``.load()`` has not been called before ``.predict()``.
        """
        if not self._is_loaded:
            raise RuntimeError(
                "InferenceEngine is not loaded. Call .load() first."
            )

        # 1. Feature engineering (adds precip_clay_interaction)
        feature_dict = engineer_features(payload)
        interaction_val = feature_dict["precip_clay_interaction"]

        # 2. Align to training column order
        df = pd.DataFrame([feature_dict])[MODEL_FEATURES]

        # 3. Preprocessing (impute → scale using fitted transforms)
        X_sc = self._scaler.transform(self._imputer.transform(df))

        # 4. Level-0: generate 3 base model predictions
        base_preds: Dict[str, float] = {}
        for name in BASE_MODEL_NAMES:
            base_preds[name] = float(
                self._base_models[name].predict(X_sc)[0]
            )

        # 5. Level-1: Ridge meta-learner
        meta_input = np.array(list(base_preds.values())).reshape(1, -1)
        final_pred = float(self._meta_model.predict(meta_input)[0])

        # 6. Extract Ridge coefficients as interpretable weights
        meta_weights = dict(
            zip(BASE_MODEL_NAMES, self._meta_model.coef_.tolist())
        )

        # 7. Generate natural-language sensitivity insight
        sensitivity_log = self._build_sensitivity_log(payload, interaction_val, base_preds)

        return PredictionResult(
            final_prediction=final_pred,
            base_predictions=base_preds,
            meta_weights=meta_weights,
            interaction_value=interaction_val,
            sensitivity_log=sensitivity_log,
        )

    # ------------------------------------------------------------------
    # Transparency helpers
    # ------------------------------------------------------------------

    @property
    def ridge_coefficients(self) -> Optional[Dict[str, float]]:
        """Return Ridge model coefficients keyed by base model name."""
        if not self._is_loaded:
            return None
        return dict(zip(BASE_MODEL_NAMES, self._meta_model.coef_.tolist()))

    @property
    def metadata(self) -> Dict:
        """Return the artefact metadata dictionary (version, features, etc.)."""
        return self._metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sensitivity_log(
        payload: InputPayload,
        interaction_val: float,
        base_preds: Dict[str, float],
    ) -> List[str]:
        """
        Produce a list of human-readable insights that explain the key
        drivers of the current prediction.

        These are deterministic rule-based annotations — no additional
        model inference is required.
        """
        log: List[str] = []

        # Interaction term magnitude
        if interaction_val > 100_000:
            log.append(
                f"⚡ High precipitation ({payload.annual_precip:.0f} mm/yr) in "
                f"clay-heavy soil ({payload.clay:.1f}%) amplified the "
                f"precip×clay interaction term to {interaction_val:,.0f} — "
                "waterlogging stress is a dominant biomass driver here."
            )
        elif interaction_val < 20_000:
            log.append(
                f"💧 Low precipitation×clay interaction ({interaction_val:,.0f}) "
                "suggests reduced waterlogging pressure; soil texture is less "
                "constraining to above-ground biomass in this scenario."
            )

        # Peatland fraction
        if payload.peat_frac > 0.7:
            log.append(
                f"🌿 High peatland fraction ({payload.peat_frac:.2f}) is "
                "characteristic of deep peat domes, which typically support "
                "high but structurally fragile above-ground biomass."
            )

        # NDVI signal
        if payload.NDVI > 0.75:
            log.append(
                f"🌱 Strong NDVI signal ({payload.NDVI:.2f}) indicates dense "
                "green canopy cover, consistent with mature forest biomass stocks."
            )
        elif payload.NDVI < 0.4:
            log.append(
                f"⚠️  Low NDVI ({payload.NDVI:.2f}) may indicate canopy degradation "
                "or seasonal senescence — model prediction carries higher uncertainty."
            )

        # Hydraulic stress
        if payload.Hydraulic_Stress > 0.8:
            log.append(
                f"🔴 Hydraulic Stress index ({payload.Hydraulic_Stress:.2f}) is "
                "elevated — root-zone moisture imbalance is likely suppressing "
                "net primary productivity and AGB accumulation."
            )

        # Model consensus
        preds_arr = np.array(list(base_preds.values()))
        spread = preds_arr.max() - preds_arr.min()
        if spread < 10:
            log.append(
                "✅ Strong base-model consensus (inter-model spread "
                f"< {spread:.1f} Mg/ha) — prediction confidence is high."
            )
        elif spread > 50:
            log.append(
                f"⚠️  High inter-model disagreement ({spread:.1f} Mg/ha spread). "
                "The Ridge meta-learner is reconciling divergent signals; "
                "consider this prediction as a probabilistic estimate."
            )

        if not log:
            log.append(
                "ℹ️  Input conditions are within typical training range — "
                "no anomalous environmental drivers detected."
            )

        return log
