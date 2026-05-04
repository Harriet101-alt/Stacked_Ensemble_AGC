# 🌿 Carbon Biomass Estimation — Stacked Ensemble Decision Support Tool

> **A production-grade Streamlit application built for the Rimba Raya tropical peat-swamp forest concession.**

---

## Project Overview

This project implements a **Stacked Ensemble Model** for predicting **Above-Ground Biomass (AGB)** in Mg/ha using eight environmental covariates derived from Google Earth Engine and satellite remote sensing.

### Model Architecture

```
8 Raw Covariates                Level-0 (Base Learners)       Level-1 (Meta-Learner)
─────────────────               ───────────────────────       ──────────────────────
Clay (%)          ─┐
Sand (%)          ─┤
Silt (%)          ─┤──► Preprocessing ──► RandomForest (RF)  ─┐
Elevation (m)     ─┤    (Impute+Scale)                         │
Precipitation     ─┼──────────────────► XGBoost (XGB)  ───────┼──► Ridge Regression ──► AGB (Mg/ha)
Peat Fraction     ─┤                                           │
Hydraulic Stress  ─┤──────────────────► SVR              ─────┘
NDVI              ─┘
              +
precip×clay (engineered)
```

**Anti-leakage design:** The Ridge meta-model is trained **exclusively on Out-of-Fold (OOF) predictions**, preventing data leakage from base models to the meta-learner.

---

## Repository Structure

```
biomass_app/
├── app.py                   # Streamlit UI — StreamlitApp class
├── inference_engine.py      # Stateless InferenceEngine + InputPayload
├── stacking_model.py        # Training pipeline (StackingOrchestrator)
├── generate_demo_models.py  # One-time script to create demo .joblib artefacts
├── requirements.txt
├── saved_models/            # Auto-created by generate_demo_models.py
│   ├── meta_model.joblib
│   ├── base_model_RF.joblib
│   ├── base_model_XGB.joblib
│   ├── base_model_SVR.joblib
│   ├── imputer.joblib
│   ├── scaler.joblib
│   └── metadata.json
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate demo model artefacts

If you have not yet trained the full model on the Rimba Raya dataset, run the
demo artefact generator to create realistic synthetic stand-in models:

```bash
python generate_demo_models.py
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

### 4. (Optional) Train on real data

Update the `DATA_PATH` in `stacking_model.py`, then:

```bash
python stacking_model.py
```

This will train the full nested cross-validation pipeline and write production
artefacts to `saved_models/`.

---

## Key Design Decisions

### OOP & Modularity

| Class | Responsibility |
|---|---|
| `BiomassDataHandler` | Data loading, feature engineering, imputation, scaling |
| `ModelTuner` | Hyperparameter search + OOF generation for a single base learner |
| `StackingOrchestrator` | Nested CV orchestration, artefact serialisation/loading |
| `InferenceEngine` | Stateless two-stage prediction pipeline (8→3→1) |
| `InputPayload` | Pydantic v2 validation with domain-specific bounds |
| `StreamlitApp` | UI lifecycle, session state, chart rendering |

### The "8-to-3" Transformation

The inference pipeline explicitly handles the mapping:
1. **8 raw covariates** → impute + scale → base model inputs
2. `precip_clay_interaction` is computed from `annual_precip × clay`
3. Each base model produces **1 prediction** → stacked into a `(1, 3)` vector
4. Ridge meta-model produces the final AGB estimate

### State Management (addressing the ipywidgets issue)

The `InferenceEngine` is loaded **once per Streamlit session** via
`@st.cache_resource`, ensuring:
- The `.models` dictionary is always populated before prediction
- No re-fitting occurs between button clicks
- Base model predictions are correctly stacked before the Ridge forward-pass

---

## Environmental Covariates

| Feature | Unit | Source | Range |
|---|---|---|---|
| Clay | % | SoilGrids ISRIC (0–5cm) | 0–100 |
| Sand | % | SoilGrids ISRIC (0–5cm) | 0–100 |
| Silt | % | SoilGrids ISRIC (0–5cm) | 0–100 |
| Elevation | m | CGIAR SRTM90 | 0–500 |
| Annual Precipitation | mm/yr | CHIRPS Daily | 500–6000 |
| Peat Fraction | 0–1 | ML Global Peatland Extent | 0–1 |
| Hydraulic Stress | — | RZM / (Precip+1) | 0–2 |
| NDVI | — | Satellite composite | -1–1 |
| precip × clay | mm·% | *Derived* | — |

---

*Built for the Rimba Raya Carbon Concession Dissertation Project.*
