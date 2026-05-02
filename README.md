# Carbon Biomass Intelligence — Rimba Raya
### Stacked Ensemble Decision Support Tool · v2.0 Deep Forest Edition

> A premium, research-grade Streamlit application for Above-Ground Biomass (AGB)
> estimation in tropical peat-swamp forests, built on a three-model stacked ensemble
> grounded in *Frontiers in Plant Science* (2025) and MPI Biomass Modelling principles.

---

## Overview

This tool provides an interactive decision support interface for estimating AGB (Mg/ha)
across the Rimba Raya concession (Central Kalimantan, Indonesia). It implements a
**two-stage stacked ensemble**: three heterogeneous base learners (RF, XGB, SVR) whose
Out-of-Fold predictions are combined by a Ridge meta-learner to produce a final
consensus estimate.

The v2.0 release adds:
- **Deep Forest glassmorphism UI** with neon glow dynamics
- **Live API telemetry** (Open-Meteo + Open-Elevation — no keys required)
- **Multi-page layout**: Dashboard + Model Story
- **Scenario Vault** for saving and comparing scenarios
- **Carbon Credit Calculator** (AGB → C → CO₂e → credits)
- **Consensus Distribution Overlay** (KDE + uncertainty bands)
- **Interactive 3D surface** of the Precip × Clay interaction term
- **Lottie animations** for ambient visual feedback

---

## Architecture

```
9 Environmental Covariates
    │
    ├── clay, sand, silt          (Soil texture, %)
    ├── elev                      (Elevation, m)
    ├── annual_precip             (mm/yr)
    ├── peat_frac                 (0–1)
    ├── Hydraulic_Stress          (root-zone moisture index)
    ├── NDVI                      (−1 to 1)
    └── precip_clay_interaction   (derived: precip × clay)
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
    [RF]   [XGB]   [SVR]        ← Level 0: Base Learners
      │       │       │
      └───────┴───────┘
              │
       [Ridge Meta-Model]        ← Level 1: Meta-Learner (OOF-trained)
              │
          AGB (Mg/ha)            ← Final consensus prediction
```

### Why a Stacked Ensemble?

| Property | RF | XGB | SVR | Stacked |
|---|---|---|---|---|
| Non-linear interactions | High | High | Medium | Best |
| Robustness to outliers | ✅ | ⚠️ | ✅ | ✅ |
| Gradient refinement | ❌ | ✅ | ❌ | ✅ via meta |
| Smooth interpolation | ⚠️ | ⚠️ | ✅ | ✅ |
| Avg. R² (demo artefacts) | 0.872 | 0.884 | 0.831 | **0.913** |

---

## Modules

| File | Role |
|---|---|
| `app.py` | Streamlit UI — Deep Forest Edition (v2.0) |
| `inference_engine.py` | Stateless two-stage prediction pipeline + `InputPayload` validation |
| `stacking_model.py` | Training orchestrator with nested cross-validation + OOF stacking |
| `generate_demo_models.py` | One-time script to create synthetic `saved_models/` for demo use |
| `requirements.txt` | Pinned Python dependencies |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Python ≥ 3.10 recommended. XGBoost requires a C++ compiler on some platforms.

### 2. Generate demo model artefacts

```bash
python generate_demo_models.py
```

This creates `saved_models/` with:
- `meta_model.joblib` — Ridge meta-learner
- `base_model_RF.joblib`, `base_model_XGB.joblib`, `base_model_SVR.joblib`
- `imputer.joblib`, `scaler.joblib`
- `metadata.json`

### 3. Launch the app

```bash
streamlit run app.py
```

---

## Feature Reference

### Dashboard Page

| Component | Description |
|---|---|
| **KPI Cards** | Stacked Prediction, Consensus Score, RF, XGB estimates |
| **Neon Glow** | Auto-activates on the Stacked card when Consensus Score > 85% |
| **Carbon Calculator** | AGB → Carbon (×0.5) → CO₂e (×3.67) → Credits/ha |
| **Consensus Overlay** | KDE distribution chart showing all 3 base models + Stacked |
| **Ridge Weights** | Horizontal bar chart of meta-model coefficients |
| **Sensitivity Log** | Rule-based natural-language driver annotations |
| **Session History** | Sparkline of last 20 predictions |
| **Scenario Vault** | Save, label, compare, and export scenarios |

### Model Story Page

| Component | Description |
|---|---|
| **Architecture Diagram** | Sankey flow from 9 features → 3 models → Ridge → AGB |
| **Performance Table** | R², RMSE, MAE for each model (demo values) |
| **Feature Importance** | Horizontal bar chart ranked by RF-proxy importances |
| **3D Surface** | Interactive AGB surface as a function of Clay × Precipitation |
| **Scientific Rationale** | Literature-grounded explanation of the interaction term |

### Live Mode

When the **Live Mode** toggle is active in the sidebar, the app fetches real-time data
for the Rimba Raya coordinates (lat −0.5, lon 112.5) from two free APIs:

- **Open-Meteo** (`api.open-meteo.com`) — hourly precipitation + soil moisture → annualised
- **Open-Elevation** (`api.open-elevation.com`) — terrain elevation

The `Elevation`, `Annual Precipitation`, and `Hydraulic Stress` sliders automatically
update to these live values. No API keys are required.

---

## Carbon Credit Formula

```
Carbon Stock  (Mg C/ha)   = AGB × 0.5
CO₂ Equivalent (Mg CO₂e/ha) = Carbon Stock × 3.67
Credits/ha                = CO₂e / 1000    (at 1 t/credit)
```

The 0.5 factor is the IPCC default biomass-to-carbon conversion.
The 3.67 factor is the molecular weight ratio CO₂/C (44/12).

---

## The Precip × Clay Interaction Term

The derived feature `precip_clay_interaction = annual_precip × clay` is the
model's most diagnostically important engineered input. It captures a physical
phenomenon well-documented in tropical peat science:

- **High clay + high precip** → persistent waterlogging → anaerobic decomposition
  → deep peat accumulation → structurally distinct AGB profile
- **Low clay (sandy) + high precip** → rapid drainage → nutrient leaching
  → reduced AGB despite identical rainfall

This interaction outranks raw precipitation alone in feature importance rankings,
consistent with findings from Page et al. (2011) and the MPI Biomass Modelling
Framework for inundation-prone biomes.

---

## Reproducibility

To retrain on the original Rimba Raya dataset:

```python
from stacking_model import BiomassDataHandler, StackingOrchestrator, build_default_configs

handler = BiomassDataHandler("path/to/rimba_raya.csv", MODEL_FEATURES, "AGB")
handler.load()
handler.engineer_features()

orchestrator = StackingOrchestrator(handler, build_default_configs())
orchestrator.run_nested_stacking(n_outer=5)
orchestrator.save_artefacts("saved_models")
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥1.35 | Web UI framework |
| scikit-learn | ≥1.4 | RF, SVR, Ridge, preprocessing |
| xgboost | ≥2.0 | Gradient-boosted base learner |
| plotly | ≥5.22 | Interactive charts + 3D surface |
| pydantic | ≥2.0 | Input validation schema |
| requests | ≥2.31 | Live API telemetry |
| scipy | any | KDE for consensus overlay |
| streamlit-lottie | ≥0.0.5 | Animated Lottie assets (optional) |
| numpy / pandas | ≥latest | Data handling |

---

## Scientific References

1. **Frontiers in Plant Science (2025)** — Tropical peat-swamp AGB modelling with
   remote sensing covariates and ensemble machine learning methods.

2. **Page, S.E. et al. (2011)** — *A record of Late Pleistocene and Holocene carbon
   accumulation and climate change from an equatorial peat bog (Kalimantan, Indonesia).*
   Journal of Quaternary Science.

3. **Hastie, T., Tibshirani, R., Friedman, J. (2009)** — *The Elements of Statistical
   Learning.* Springer.

4. **IPCC (2006)** — *2006 IPCC Guidelines for National Greenhouse Gas Inventories,
   Vol. 4: Agriculture, Forestry and Other Land Use.*

5. **MPI Biomass Modelling Principles** — Internal framework for biomass estimation
   in carbon-dense tropical ecosystems.

---

*Built with Streamlit · Scikit-learn · XGBoost · Plotly · Open-Meteo · Open-Elevation*
