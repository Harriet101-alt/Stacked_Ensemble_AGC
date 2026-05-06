"""
app.py  ·  v5.0 — Futuristic Emerald Edition
==============================================
Carbon Biomass Stacked Ensemble — Decision Support Tool

Clean Neo-Minimalism (Tremor.so) with futuristic dark modules.
CVD-safe palette · Double-encoded charts · WCAG AA compliant

Multi-page layout:
  DASHBOARD   — Hero, Landing, Live inference, KPI cards, Carbon Credits
  MODEL STORY — Ensemble lineage, Feature Importance, Precip × Clay science

Run:  streamlit run app.py

Author : Harriet Fletcher
"""

from __future__ import annotations

import base64
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pydantic import ValidationError

from inference_engine import InferenceEngine, InputPayload, PredictionResult

# ─────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rimba Raya · Biomass Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "Images")
PLOTLY_TEMPLATE = "simple_white"
RIMBA_RAYA_LAT = -0.5
RIMBA_RAYA_LON = 112.5

# ── Colour Palette — Professional Light Theme ────────────────
COLORS = {
    # Model identity (qualitative, CVD-safe)
    "RF":       "#377eb8",   # Blue
    "XGB":      "#ff7f00",   # Orange
    "SVR":      "#984ea3",   # Purple
    "Stacked":  "#10B981",   # Emerald (primary)

    # Semantic
    "positive":  "#10B981",  # Emerald — consensus / good
    "warning":   "#F59E0B",  # Amber — caution
    "alert":     "#EF4444",  # Red — high-risk regime

    # UI chrome (Tremor-inspired)
    "primary":      "#10B981",
    "primary_dark": "#059669",
    "bg":           "#F9FAFB",
    "card_bg":      "#FFFFFF",
    "card_border":  "#E5E7EB",

    # Typography (dark text on light bg — all >> 4.5:1 contrast)
    "text_primary": "#111827",
    "text_body":    "#374151",
    "text_mid":     "#4B5563",
    "text_muted":   "#6B7280",

    # Carbon credits
    "carbon_gold": "#F59E0B",
}

# ── Double-Encoding — line style + marker per model ──────────
MODEL_STYLES = {
    "RF":      {"dash": "solid", "marker": "circle",  "pattern": "/",  "width": 2},
    "XGB":     {"dash": "dash",  "marker": "square",  "pattern": "\\", "width": 2},
    "SVR":     {"dash": "dot",   "marker": "diamond", "pattern": "x",  "width": 2},
    "Stacked": {"dash": "solid", "marker": "star",    "pattern": "",   "width": 3.5},
}

# ── Plotly defaults ──────────────────────────────────────────
PLOTLY_FONT = dict(family="Inter, sans-serif", color="#111827")

# ── Emerald-to-Slate color scale for charts ──────────────────
EMERALD_SLATE_SCALE = [
    [0, "#064E3B"],
    [0.25, "#059669"],
    [0.5, "#10B981"],
    [0.75, "#6EE7B7"],
    [1, "#E2E8F0"],
]


# ── Safe hex-to-rgba conversion ──────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ── Image loading helper (Base64 for CSS injection) ──────────
@st.cache_data(show_spinner=False)
def _load_image_b64(filename: str) -> Optional[str]:
    """Load an image from IMAGE_DIR and return a Base64 data URI."""
    img_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }.get(ext, "image/png")
    return f"data:{mime};base64,{data}"


# ─────────────────────────────────────────────────────────────
# CSS — Neo-Minimalist Light Theme + Futuristic Dark Modules
# ─────────────────────────────────────────────────────────────
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── App Background — soft off-white ── */
.stApp {
    background-color: #F9FAFB;
    color: #111827;
}

/* ── Sidebar — clean white ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}

[data-testid="stSidebar"] .stSlider label {
    color: #374151 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* Slider track — emerald */
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: linear-gradient(90deg, #10B981, #34D399) !important;
}

/* ── Tremor card — white with light border, 12px radius ── */
.tremor-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

/* ── KPI card ── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 22px 20px;
    text-align: center;
    transition: box-shadow 0.2s ease;
}
.kpi-card:hover {
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

.kpi-card.consensus-glow {
    border-color: #10B981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25),
                0 0 16px rgba(16, 185, 129, 0.20),
                0 0 32px rgba(16, 185, 129, 0.10);
}

.kpi-label {
    color: #6B7280;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Inter', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
}
.kpi-unit {
    color: #6B7280;
    font-size: 0.72rem;
    margin-top: 6px;
}

/* ── Section title ── */
.section-title {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4B5563;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── Log entries ── */
.log-entry {
    background: #F0FDF4;
    border-left: 3px solid #10B981;
    border-radius: 0 12px 12px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.55;
}

/* ── Interaction term chip ── */
.interaction-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #6B7280;
}
.interaction-chip code {
    color: #10B981;
    background: transparent;
    font-weight: 600;
}

/* ── Run button — emerald ── */
.stButton > button {
    background: #10B981 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1.4rem !important;
    width: 100% !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover {
    background: #059669 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #E5E7EB !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
    background: transparent !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 8px 18px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #10B981 !important;
    border-bottom: 2px solid #10B981 !important;
}

/* ── Carbon card — warm amber tint ── */
.carbon-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 12px;
    padding: 18px 22px;
}

/* ── Sidebar — force all text visible on white bg ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown {
    color: #4B5563 !important;
}

/* ── Radio nav — charcoal text ── */
div[data-testid="stRadio"] label,
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: #374151 !important;
}

/* ── Toggle ── */
div[data-testid="stToggle"] label,
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    font-size: 0.82rem !important;
    color: #6B7280 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* ── Live badge ── */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FEF3C7;
    border: 1px solid #FDE68A;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #B45309;
}

/* ── HR divider ── */
hr { border-color: #E5E7EB !important; margin: 1.2rem 0 !important; }

/* ── Metric override ── */
div[data-testid="stMetricValue"] { color: #10B981 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F9FAFB; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* ── Futuristic dark module (Precip x Clay, Hero accents) ── */
.dark-module {
    background: #0A0A0A;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 12px;
}

.dark-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Live API helpers (Open-Meteo + Open-Elevation)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_live_telemetry(lat: float, lon: float) -> dict:
    """Pull real-time precipitation, soil moisture, and elevation."""
    result = {}
    try:
        meteo_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=precipitation,soil_moisture_0_to_1cm"
            f"&timezone=auto&forecast_days=1"
        )
        r = requests.get(meteo_url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            precip_vals = data.get("hourly", {}).get("precipitation", [0] * 24)
            soil_vals = data.get("hourly", {}).get("soil_moisture_0_to_1cm", [0.25] * 24)
            mean_hourly_precip = np.nanmean(
                [v for v in precip_vals if v is not None] or [0]
            )
            result["annual_precip"] = float(np.clip(mean_hourly_precip * 8760, 500, 6000))
            mean_soil = np.nanmean(
                [v for v in soil_vals if v is not None] or [0.25]
            )
            result["soil_moisture"] = float(mean_soil)
            result["Hydraulic_Stress"] = float(np.clip(1.0 - mean_soil, 0, 2))
    except Exception:
        pass

    try:
        elev_url = (
            f"https://api.open-elevation.com/api/v1/lookup"
            f"?locations={lat},{lon}"
        )
        r = requests.get(elev_url, timeout=8)
        if r.status_code == 200:
            elev_data = r.json()
            elev = elev_data.get("results", [{}])[0].get("elevation", 5)
            result["elev"] = float(np.clip(elev, 0, 500))
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────
# Soil texture slider callbacks — keep clay+sand+silt == 100
# ─────────────────────────────────────────────────────────────

def _redistribute_soil(changed_key: str) -> None:
    keys = ["_soil_clay", "_soil_sand", "_soil_silt"]
    other_keys = [k for k in keys if k != changed_key]
    remainder = 100.0 - st.session_state[changed_key]
    a = st.session_state[other_keys[0]]
    b = st.session_state[other_keys[1]]
    total_other = a + b
    if total_other == 0:
        st.session_state[other_keys[0]] = round(remainder / 2, 1)
        st.session_state[other_keys[1]] = round(remainder / 2, 1)
    else:
        new_a = round(a / total_other * remainder, 1)
        st.session_state[other_keys[0]] = new_a
        st.session_state[other_keys[1]] = round(remainder - new_a, 1)


def _on_clay_change() -> None:
    _redistribute_soil("_soil_clay")


def _on_sand_change() -> None:
    _redistribute_soil("_soil_sand")


def _on_silt_change() -> None:
    _redistribute_soil("_soil_silt")


# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        "prediction_result": None,
        "prediction_history": [],
        "scenario_vault": [],
        "active_page": "DASHBOARD",
        "live_mode": False,
        "live_telemetry": {},
        "_last_payload_hash": None,
        "_prev_live_mode": False,
        "_soil_clay": 35.0,
        "_soil_sand": 40.0,
        "_soil_silt": 25.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# Engine loader (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_engine() -> InferenceEngine:
    engine = InferenceEngine(MODEL_DIR)
    engine.load()
    return engine


# ─────────────────────────────────────────────────────────────
# Main Application Class
# ─────────────────────────────────────────────────────────────

class PremiumBiomassApp:
    def __init__(self) -> None:
        _init_session_state()
        self.engine: InferenceEngine = _load_engine()

    # ── Entry point ───────────────────────────────────────────
    def run(self) -> None:
        payload, run_clicked = self._render_sidebar()

        # Clear stale prediction when slider values change
        if payload is not None:
            current_hash = hash(
                tuple(sorted(payload.model_dump().items()))
            )
            if (
                st.session_state["_last_payload_hash"] is not None
                and st.session_state["_last_payload_hash"] != current_hash
            ):
                st.session_state["prediction_result"] = None
        else:
            st.session_state["prediction_result"] = None

        self._render_hero()
        page = st.session_state["active_page"]
        if page == "DASHBOARD":
            self._page_dashboard(payload, run_clicked)
        else:
            self._page_model_story()

    # ── Hero Banner (Trees.png) ───────────────────────────────
    def _render_hero(self) -> None:
        hero_b64 = _load_image_b64("Trees.png")
        if hero_b64:
            bg_css = (
                f"background-image: linear-gradient(rgba(0,0,0,0.20),"
                f" rgba(0,0,0,0.25)), url('{hero_b64}');"
                f" background-size: cover; background-position: center;"
            )
        else:
            bg_css = "background: linear-gradient(135deg, #059669 0%, #10B981 100%);"
        st.markdown(
            f"<style>.hero-bg {{ {bg_css} }}</style>"
            "<div class='hero-bg' style='height: 300px; border-radius: 12px;"
            " display: flex; flex-direction: column; justify-content: center;"
            " align-items: center; margin-bottom: 24px; overflow: hidden;'>"
            "<h1 style='color: #FFFFFF; font-family: Inter, sans-serif;"
            " font-size: 2.4rem; font-weight: 700; margin: 0;"
            " text-shadow: 0 2px 12px rgba(0,0,0,0.5);'>"
            "Rimba Raya Biomass Intelligence</h1>"
            "<p style='color: rgba(255,255,255,0.95); font-family: Inter, sans-serif;"
            " font-size: 1.05rem; margin-top: 8px; font-weight: 400;"
            " text-shadow: 0 1px 6px rgba(0,0,0,0.4);'>"
            "Stacked Ensemble &middot; Carbon Quantification"
            " &middot; Decision Support</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════════════════════════
    def _render_sidebar(self) -> tuple[Optional[InputPayload], bool]:
        with st.sidebar:
            st.markdown(
                "<div style='font-family: Inter, sans-serif; font-size: 1.05rem;"
                f" font-weight: 700; color: {COLORS['primary']};"
                " letter-spacing: 0.02em; margin: 4px 0 2px;'>"
                "RIMBA RAYA</div>"
                "<div style='font-size: 0.68rem;"
                f" color: {COLORS['text_muted']}; letter-spacing: 0.08em;"
                " text-transform: uppercase; margin-bottom: 12px;'>"
                "Biomass Intelligence &middot; v5.0</div>",
                unsafe_allow_html=True,
            )

            # ── Page navigation ────────────────────────────────
            st.radio(
                "Navigate",
                ["DASHBOARD", "MODEL STORY"],
                key="active_page",
                label_visibility="collapsed",
                horizontal=True,
            )

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Live Mode toggle ───────────────────────────────
            live_mode = st.toggle(
                "Live Mode  (Rimba Raya telemetry)",
                value=st.session_state["live_mode"],
                key="live_mode",
            )

            # Clear stale state when switching modes
            if live_mode != st.session_state.get("_prev_live_mode", False):
                st.session_state["prediction_result"] = None
                st.session_state["_last_payload_hash"] = None
                if not live_mode:
                    st.session_state["live_telemetry"] = {}
                st.session_state["_prev_live_mode"] = live_mode

            if live_mode:
                st.markdown(
                    "<div class='live-badge'>LIVE  |  lat -0.5  lon 112.5</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Fetching telemetry..."):
                    telem = _fetch_live_telemetry(
                        RIMBA_RAYA_LAT, RIMBA_RAYA_LON
                    )
                    st.session_state["live_telemetry"] = telem
                if telem:
                    st.caption(
                        f"Live: Precip est. {telem.get('annual_precip', 0):.0f}"
                        f" mm/yr | Elev {telem.get('elev', 0):.0f} m"
                    )
                else:
                    st.caption("API unreachable — using slider values")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Sliders — grouped by domain ────────────────────
            telem = (
                st.session_state.get("live_telemetry", {})
                if live_mode else {}
            )

            with st.expander("Soil Texture", expanded=True):
                clay = st.slider(
                    "Clay (%)", 0.0, 100.0,
                    step=0.5,
                    key="_soil_clay",
                    on_change=_on_clay_change,
                    help="Percentage of clay particles in the topsoil. "
                         "Higher clay increases water retention and "
                         "waterlogging risk.",
                )
                sand = st.slider(
                    "Sand (%)", 0.0, 100.0,
                    step=0.5,
                    key="_soil_sand",
                    on_change=_on_sand_change,
                    help="Percentage of sand in the topsoil. "
                         "Sandy soils drain quickly and retain fewer "
                         "nutrients.",
                )
                silt = st.slider(
                    "Silt (%)", 0.0, 100.0,
                    step=0.5,
                    key="_soil_silt",
                    on_change=_on_silt_change,
                    help="Percentage of silt in the topsoil. "
                         "Silt contributes to fertility and moderate "
                         "drainage.",
                )
                _soil_total = round(
                    st.session_state["_soil_clay"]
                    + st.session_state["_soil_sand"]
                    + st.session_state["_soil_silt"]
                )
                if _soil_total == 100:
                    st.caption(
                        "<span style='color:#10B981'>Total: 100%</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(
                        f"<span style='color:#F59E0B'>"
                        f"Total: {_soil_total}% ⚠️</span>",
                        unsafe_allow_html=True,
                    )

            with st.expander("Hydrology & Topography", expanded=True):
                elev_default = float(telem.get("elev", 12.0))
                precip_default = float(telem.get("annual_precip", 2800.0))
                hyd_default = float(telem.get("Hydraulic_Stress", 0.25))

                elev = st.slider(
                    "Elevation (m)", 0.0, 500.0, elev_default, 1.0,
                    help="Height above sea level. Lower elevations in "
                         "peat-swamp regions tend to accumulate more "
                         "biomass.",
                )
                precip = st.slider(
                    "Annual Precipitation (mm/yr)",
                    500.0, 6000.0, precip_default, 50.0,
                    help="Total yearly rainfall. Interacts with clay "
                         "content to determine waterlogging intensity.",
                )
                peat_frac = st.slider(
                    "Peat Fraction (0-1)", 0.0, 1.0, 0.75, 0.01,
                    help="Proportion of peatland in the area. Deep peat "
                         "domes support high but structurally fragile "
                         "biomass.",
                )
                hyd_stress = st.slider(
                    "Hydraulic Stress Index", 0.0, 2.0, hyd_default, 0.01,
                    help="Ratio of root-zone moisture demand to supply. "
                         "Higher values indicate moisture deficit stress.",
                )

            with st.expander("Vegetation", expanded=True):
                ndvi = st.slider(
                    "NDVI", -1.0, 1.0, 0.78, 0.01,
                    help="Normalised Difference Vegetation Index — a "
                         "satellite-derived measure of canopy greenness "
                         "(range -1 to 1).",
                )

            st.markdown("<hr>", unsafe_allow_html=True)

            # Interaction term preview
            interaction_preview = precip * clay
            st.markdown(
                "<div class='tremor-card' style='padding: 12px 16px;"
                " margin-bottom: 12px;'>"
                f"<div style='font-size: 0.68rem; color: {COLORS['text_muted']};"
                " text-transform: uppercase; letter-spacing: 0.08em;'>"
                "Precip &times; Clay</div>"
                f"<div style='font-size: 1.5rem; font-weight: 700;"
                f" color: {COLORS['primary']}; margin-top: 2px;'>"
                f"{interaction_preview:,.0f}</div>"
                f"<div style='font-size: 0.68rem; color: {COLORS['text_muted']};'>"
                "mm &middot; %</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            run_clicked = st.button(
                "Run Ensemble Inference", type="primary"
            )

        # Validate payload
        payload: Optional[InputPayload] = None
        try:
            payload = InputPayload(
                clay=clay, sand=sand, silt=silt, elev=elev,
                annual_precip=precip, peat_frac=peat_frac,
                Hydraulic_Stress=hyd_stress, NDVI=ndvi,
            )
        except ValidationError as exc:
            if run_clicked:
                st.sidebar.error(
                    f"Validation: {exc.errors()[0]['msg']}"
                )

        return payload, run_clicked

    # ════════════════════════════════════════════════════════
    # PAGE: DASHBOARD
    # ════════════════════════════════════════════════════════
    def _page_dashboard(
        self,
        payload: Optional[InputPayload],
        run_clicked: bool,
    ) -> None:
        # ── Landing Section ────────────────────────────────────
        self._render_landing()

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Section header ─────────────────────────────────────
        st.markdown(
            "<div style='display: flex; align-items: baseline; gap: 12px;"
            " margin-bottom: 4px;'>"
            f"<span style='font-family: Inter, sans-serif; font-size: 1.6rem;"
            f" font-weight: 700; color: {COLORS['text_primary']};'>"
            "Carbon Biomass Intelligence</span>"
            f"<span style='font-size: 0.7rem; color: {COLORS['text_muted']};"
            " letter-spacing: 0.08em; text-transform: uppercase;"
            " padding: 3px 10px; border: 1px solid #E5E7EB;"
            " border-radius: 20px;'>RF &middot; XGB &middot; SVR"
            " &rarr; Ridge</span>"
            "</div>"
            f"<div style='font-size: 0.78rem; color: {COLORS['text_mid']};"
            " margin-bottom: 20px;'>"
            "Stacked Ensemble &nbsp;&middot;&nbsp;"
            " Rimba Raya Peat-Swamp Forest &nbsp;&middot;&nbsp;"
            " Frontiers in Plant Science (2025) &nbsp;&middot;&nbsp;"
            " MPI Biomass Principles"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Run inference ──────────────────────────────────────
        if run_clicked and payload is not None:
            placeholder = st.empty()
            with placeholder.container():
                st.markdown(
                    "<div style='text-align: center; padding: 24px 0;"
                    f" color: {COLORS['text_mid']}; font-size: 0.82rem;"
                    " letter-spacing: 0.08em;'>"
                    "RUNNING TWO-STAGE ENSEMBLE...</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.6)
            placeholder.empty()

            try:
                result: PredictionResult = self.engine.predict(payload)
                st.session_state["prediction_result"] = result
                st.session_state["_last_payload_hash"] = hash(
                    tuple(sorted(payload.model_dump().items()))
                )
                st.session_state["prediction_history"].append({
                    "inputs": payload.model_dump(),
                    "final": result.final_prediction,
                    "base": result.base_predictions,
                })
                if len(st.session_state["prediction_history"]) > 20:
                    st.session_state["prediction_history"].pop(0)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")

        result: Optional[PredictionResult] = st.session_state[
            "prediction_result"
        ]

        # ── KPI Row ────────────────────────────────────────────
        self._render_kpi_row(result)
        st.markdown("")

        # ── Carbon Credit Calculator ───────────────────────────
        if result:
            self._render_carbon_calculator(result)
            st.markdown("")

        # ── Main charts ────────────────────────────────────────
        tab_charts, tab_vault = st.tabs(
            ["ANALYSIS", "SCENARIO VAULT"]
        )

        with tab_charts:
            col_left, col_right = st.columns([3, 2], gap="large")
            with col_left:
                self._render_consensus_overlay(result)
            with col_right:
                self._render_ridge_transparency(result)

            st.markdown("")
            col_log, col_hist = st.columns([3, 2], gap="large")
            with col_log:
                self._render_sensitivity_log(result)
            with col_hist:
                self._render_prediction_history()

        with tab_vault:
            self._render_scenario_vault(result, payload)

    # ── Landing Section (Map + Intro) ─────────────────────────
    def _render_landing(self) -> None:
        col_text, col_map = st.columns([3, 2], gap="large")

        with col_text:
            st.markdown(
                "<div class='tremor-card'>"
                f"<h3 style='color: {COLORS['text_primary']}; margin-top: 0;"
                " font-size: 1.2rem; font-weight: 700;'>"
                "Rimba Raya Biodiversity Reserve</h3>"
                f"<p style='color: {COLORS['text_body']}; line-height: 1.75;"
                " font-size: 0.92rem; margin-bottom: 14px;'>"
                "The <strong>Rimba Raya Biodiversity Reserve</strong> spans"
                " over 64,000 hectares of tropical peat-swamp forest in"
                " Central Kalimantan, Indonesian Borneo. Situated between"
                " the Java Sea and Tanjung Puting National Park, this"
                " critically important ecosystem sits atop deep peat"
                " deposits that have accumulated organic carbon over"
                " millennia &mdash; making it one of Earth's most"
                " carbon-dense terrestrial biomes."
                "</p>"
                f"<p style='color: {COLORS['text_body']}; line-height: 1.75;"
                " font-size: 0.92rem; margin-bottom: 14px;'>"
                "Accurate quantification of above-ground biomass (AGB)"
                " underpins the credibility of nature-based carbon markets."
                " Under frameworks such as <strong>REDD+</strong> and"
                " <strong>Verra's Verified Carbon Standard (VCS)</strong>,"
                " each verified tonne of CO&#8322; equivalent sequestered in"
                " standing forest biomass can be issued as a tradeable"
                " carbon credit. The integrity of these credits &mdash;"
                " and by extension, the billions of dollars invested in"
                " voluntary carbon markets &mdash; depends on the precision,"
                " transparency, and scientific rigour of the underlying"
                " estimation models."
                "</p>"
                f"<p style='color: {COLORS['text_body']}; line-height: 1.75;"
                " font-size: 0.92rem;'>"
                "This tool applies a <strong>stacked ensemble approach</strong>,"
                " combining three heterogeneous base learners (Random Forest,"
                " XGBoost, and SVR) through a Ridge regression meta-learner,"
                " to produce robust, auditable AGB predictions grounded in"
                " peer-reviewed methodology and satellite-derived"
                " environmental covariates."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_map:
            map_path = os.path.join(IMAGE_DIR, "Map.png")
            if os.path.exists(map_path):
                st.markdown(
                    "<div class='tremor-card' style='padding: 12px;"
                    " overflow: hidden; text-align: center;'>",
                    unsafe_allow_html=True,
                )
                st.image(
                    map_path,
                    width='stretch',
                    caption="Rimba Raya, Central Kalimantan, Indonesia",
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────
    def _render_kpi_row(
        self, result: Optional[PredictionResult]
    ) -> None:
        col1, col2, col3, col4 = st.columns(4, gap="medium")

        if result:
            final = result.final_prediction
            bp = result.base_predictions
            rf_p = bp.get("RF", 0)
            xgb_p = bp.get("XGB", 0)
            svr_p = bp.get("SVR", 0)
            spread = max(rf_p, xgb_p, svr_p) - min(rf_p, xgb_p, svr_p)
            confidence = max(0.0, 100.0 - spread * 0.5)
            high_consensus = confidence > 85
        else:
            final = rf_p = xgb_p = svr_p = None
            confidence = None
            high_consensus = False

        glow_class = (
            "kpi-card consensus-glow" if high_consensus else "kpi-card"
        )

        with col1:
            st.markdown(
                f"<div class='{glow_class}'>"
                "<div class='kpi-label'>STACKED PREDICTION</div>"
                f"<div class='kpi-value' style='color: {COLORS['Stacked']};"
                f" font-size: 3.2rem;'>"
                f"{f'{final:.1f}' if final else '&mdash;'}</div>"
                "<div class='kpi-unit'>Mg / ha &middot; AGB</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with col2:
            if confidence is not None:
                if confidence > 75:
                    c_color = COLORS["positive"]
                    c_icon = "&#10003;"
                    c_desc = "Strong model agreement"
                else:
                    c_color = COLORS["warning"]
                    c_icon = "&#9888;"
                    c_desc = "Models show disagreement"
            else:
                c_color = COLORS["text_muted"]
                c_icon = "&mdash;"
                c_desc = "Inter-model agreement"
            st.markdown(
                "<div class='kpi-card'>"
                "<div class='kpi-label'>CONSENSUS SCORE</div>"
                f"<div class='kpi-value' style='color: {c_color};'>"
                f"{c_icon} "
                f"{f'{confidence:.0f}%' if confidence is not None else '&mdash;'}"
                "</div>"
                f"<div class='kpi-unit'>{c_desc}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                "<div class='kpi-card'>"
                "<div class='kpi-label'>RF ESTIMATE</div>"
                f"<div class='kpi-value' style='color: {COLORS['RF']};'>"
                f"{f'{rf_p:.1f}' if rf_p else '&mdash;'}</div>"
                "<div class='kpi-unit'>Mg / ha</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                "<div class='kpi-card'>"
                "<div class='kpi-label'>XGB ESTIMATE</div>"
                f"<div class='kpi-value' style='color: {COLORS['XGB']};'>"
                f"{f'{xgb_p:.1f}' if xgb_p else '&mdash;'}</div>"
                "<div class='kpi-unit'>Mg / ha</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Carbon Credit Calculator ──────────────────────────────
    def _render_carbon_calculator(
        self, result: PredictionResult
    ) -> None:
        agb = result.final_prediction
        carbon_stock = agb * 0.5
        co2e = carbon_stock * 3.67
        credits = co2e / 1000

        col1, col2, col3, col4 = st.columns(4, gap="medium")
        items = [
            ("AGB Stock", f"{agb:.1f}", "Mg/ha",
             COLORS["Stacked"]),
            ("Carbon Stock", f"{carbon_stock:.1f}",
             "Mg C/ha  (&times;0.5)", COLORS["positive"]),
            ("CO2 Equivalent", f"{co2e:.1f}",
             "Mg CO2e/ha  (&times;3.67)", COLORS["SVR"]),
            ("Carbon Credits", f"{credits:.2f}",
             "Credits/ha  (@ 1 t/credit)", COLORS["carbon_gold"]),
        ]
        for col, (label, val, unit, color) in zip(
            [col1, col2, col3, col4], items
        ):
            with col:
                st.markdown(
                    "<div class='carbon-card'>"
                    "<div style='font-size: 0.65rem; color: #92400E;"
                    " text-transform: uppercase; letter-spacing: 0.08em;'>"
                    f"{label}</div>"
                    f"<div style='font-size: 1.9rem; font-weight: 700;"
                    f" color: {color}; margin: 4px 0;'>{val}</div>"
                    f"<div style='font-size: 0.65rem; color: #92400E;'>"
                    f"{unit}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── Consensus Overlay Chart ───────────────────────────────
    def _render_consensus_overlay(
        self, result: Optional[PredictionResult]
    ) -> None:
        st.markdown(
            "<p class='section-title'>Consensus Distribution Overlay</p>",
            unsafe_allow_html=True,
        )

        if result is None:
            st.markdown(
                f"<div style='color: {COLORS['text_muted']};"
                " font-size: 0.9rem; padding: 40px 0;"
                " text-align: center;'>"
                "Run inference to populate charts.</div>",
                unsafe_allow_html=True,
            )
            return

        bp = result.base_predictions
        final = result.final_prediction

        rng = np.random.default_rng(42)
        sigma = 18
        model_samples = {
            m: rng.normal(v, sigma, 800) for m, v in bp.items()
        }
        stacked_samples = rng.normal(final, sigma * 0.55, 800)

        fig = go.Figure()
        all_vals = np.concatenate(
            list(model_samples.values()) + [stacked_samples]
        )
        x_range = np.linspace(
            all_vals.min() - 10, all_vals.max() + 10, 300
        )

        from scipy.stats import gaussian_kde

        for model_name, samples in model_samples.items():
            style = MODEL_STYLES[model_name]
            color = COLORS[model_name]
            kde = gaussian_kde(samples)
            y_kde = kde(x_range)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_kde,
                mode="lines",
                name=model_name,
                line=dict(
                    color=color,
                    width=style["width"],
                    dash=style["dash"],
                ),
                fill="tozeroy",
                fillcolor=hex_to_rgba(color, 0.08),
                opacity=0.85,
            ))

        style_s = MODEL_STYLES["Stacked"]
        kde_s = gaussian_kde(stacked_samples)
        y_s = kde_s(x_range)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_s,
            mode="lines",
            name="Stacked",
            line=dict(
                color=COLORS["Stacked"],
                width=style_s["width"],
                dash=style_s["dash"],
            ),
            fill="tozeroy",
            fillcolor=hex_to_rgba(COLORS["Stacked"], 0.10),
        ))

        for model_name, val in {**bp, "Stacked": final}.items():
            style = MODEL_STYLES[model_name]
            fig.add_vline(
                x=val,
                line_color=COLORS.get(model_name, "#888"),
                line_width=1.5,
                line_dash=style["dash"],
            )

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            font=PLOTLY_FONT,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=20, l=10, r=10),
            height=300,
            legend=dict(
                orientation="h", y=1.15,
                font=dict(size=10, color=COLORS["text_primary"]),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                title="AGB (Mg/ha)",
                gridcolor="rgba(0,0,0,0.06)",
                title_font_color=COLORS["text_primary"],
                color=COLORS["text_primary"],
            ),
            yaxis=dict(
                title="Density",
                gridcolor="rgba(0,0,0,0.06)",
                title_font_color=COLORS["text_primary"],
                color=COLORS["text_primary"],
            ),
        )
        st.plotly_chart(
            fig, width='stretch',
            config={"displayModeBar": False},
        )

    # ── Ridge Transparency ────────────────────────────────────
    def _render_ridge_transparency(
        self, result: Optional[PredictionResult]
    ) -> None:
        st.markdown(
            "<p class='section-title'>Ridge Meta-Model Weights</p>",
            unsafe_allow_html=True,
        )

        if result is None:
            st.markdown(
                f"<div style='color: {COLORS['text_muted']};"
                " font-size: 0.9rem; padding: 40px 0;"
                " text-align: center;'>"
                "Run inference to reveal weights.</div>",
                unsafe_allow_html=True,
            )
            return

        weights = result.meta_weights
        names = list(weights.keys())
        coefs = list(weights.values())
        colours = [COLORS.get(n, COLORS["primary"]) for n in names]
        patterns = [
            MODEL_STYLES.get(n, {}).get("pattern", "") for n in names
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=coefs, y=names, orientation="h",
            marker=dict(
                color=colours,
                pattern=dict(
                    shape=patterns,
                    fgcolor="rgba(0,0,0,0.15)",
                    size=6,
                ),
            ),
            text=[f"{c:+.4f}" for c in coefs],
            textposition="outside",
            textfont=dict(
                size=11, color=COLORS["text_primary"],
                family="Inter",
            ),
        ))
        fig.add_vline(
            x=0, line_color="rgba(0,0,0,0.08)", line_width=1
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            font=PLOTLY_FONT,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=10, r=60),
            height=220,
            showlegend=False,
            xaxis=dict(
                title="Ridge Coeff.",
                gridcolor="rgba(0,0,0,0.06)",
                title_font_color=COLORS["text_primary"],
                color=COLORS["text_primary"],
                zeroline=False,
            ),
            yaxis=dict(
                tickfont=dict(
                    size=12, color=COLORS["text_primary"],
                    family="Inter",
                ),
            ),
        )
        st.plotly_chart(
            fig, width='stretch',
            config={"displayModeBar": False},
        )

        dominant = max(weights, key=lambda k: abs(weights[k]))
        st.caption(
            f"Ridge assigns highest absolute weight to **{dominant}** "
            f"({weights[dominant]:+.4f}). Negative coefficients indicate "
            f"corrective suppression of over-predictions."
        )

    # ── Sensitivity Log ───────────────────────────────────────
    def _render_sensitivity_log(
        self, result: Optional[PredictionResult]
    ) -> None:
        st.markdown(
            "<p class='section-title'>"
            "Sensitivity Analysis &mdash; Drivers</p>",
            unsafe_allow_html=True,
        )

        if result is None:
            st.markdown(
                f"<div style='color: {COLORS['text_muted']};"
                " font-size: 0.9rem;'>"
                "Adjust sliders and run inference to see driver"
                " analysis.</div>",
                unsafe_allow_html=True,
            )
            return

        for entry in result.sensitivity_log:
            st.markdown(
                f"<div class='log-entry'>{entry}</div>",
                unsafe_allow_html=True,
            )

        interaction = result.interaction_value
        tier = (
            "High" if interaction > 100_000
            else "Moderate" if interaction > 50_000
            else "Low"
        )
        t_color = (
            COLORS["alert"] if tier == "High"
            else COLORS["warning"] if tier == "Moderate"
            else COLORS["positive"]
        )
        t_icon = "&#9888;" if tier in ("High", "Moderate") else "&#10003;"
        st.markdown(
            "<div class='interaction-chip'>"
            f"precip &times; clay = <code>{interaction:,.0f}</code>"
            f"&nbsp;|&nbsp; Regime: "
            f"<strong style='color: {t_color};'>{t_icon} {tier}</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Prediction History ────────────────────────────────────
    def _render_prediction_history(self) -> None:
        st.markdown(
            "<p class='section-title'>Session History (Last 20)</p>",
            unsafe_allow_html=True,
        )

        history = st.session_state.get("prediction_history", [])
        if not history:
            st.markdown(
                f"<div style='color: {COLORS['text_muted']};"
                " font-size: 0.9rem;'>No history yet.</div>",
                unsafe_allow_html=True,
            )
            return

        finals = [h["final"] for h in history]
        rf_hist = [h["base"].get("RF", 0) for h in history]
        xgb_hist = [h["base"].get("XGB", 0) for h in history]
        svr_hist = [h["base"].get("SVR", 0) for h in history]

        fig = go.Figure()
        for name, vals, color in [
            ("RF", rf_hist, COLORS["RF"]),
            ("XGB", xgb_hist, COLORS["XGB"]),
            ("SVR", svr_hist, COLORS["SVR"]),
            ("Stacked", finals, COLORS["Stacked"]),
        ]:
            style = MODEL_STYLES[name]
            fig.add_trace(go.Scatter(
                y=vals, mode="lines+markers", name=name,
                line=dict(
                    color=color,
                    width=style["width"],
                    dash=style["dash"],
                ),
                marker=dict(
                    size=7 if name == "Stacked" else 5,
                    symbol=style["marker"],
                ),
                opacity=1.0 if name == "Stacked" else 0.75,
            ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            font=PLOTLY_FONT,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=240,
            legend=dict(
                orientation="h", y=1.15,
                font=dict(size=9, color=COLORS["text_primary"]),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0.06)",
                showticklabels=False,
            ),
            yaxis=dict(
                title="AGB (Mg/ha)",
                gridcolor="rgba(0,0,0,0.06)",
                title_font_color=COLORS["text_primary"],
                color=COLORS["text_primary"],
            ),
        )
        st.plotly_chart(
            fig, width='stretch',
            config={"displayModeBar": False},
        )

        if len(finals) >= 2:
            delta = finals[-1] - finals[-2]
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            st.caption(
                f"Latest: **{finals[-1]:.1f} Mg/ha** · "
                f"Delta: **{delta_str}** · "
                f"Mean: **{np.mean(finals):.1f} Mg/ha**"
            )

    # ── Scenario Vault ────────────────────────────────────────
    def _render_scenario_vault(
        self,
        result: Optional[PredictionResult],
        payload: Optional[InputPayload],
    ) -> None:
        st.markdown(
            "<p class='section-title'>"
            "Scenario Vault &mdash; Save &amp; Compare Results</p>",
            unsafe_allow_html=True,
        )

        col_save, col_clear = st.columns([1, 1])
        with col_save:
            label = st.text_input(
                "Scenario label",
                placeholder="e.g. High peat, wet season",
                label_visibility="collapsed",
            )
            if st.button(
                "Save Scenario",
                disabled=(result is None or payload is None),
            ):
                entry = {
                    "label": (
                        label
                        or f"Scenario "
                           f"{len(st.session_state['scenario_vault']) + 1}"
                    ),
                    "final": result.final_prediction,
                    "RF": result.base_predictions.get("RF", 0),
                    "XGB": result.base_predictions.get("XGB", 0),
                    "SVR": result.base_predictions.get("SVR", 0),
                    **payload.model_dump(),
                }
                st.session_state["scenario_vault"].append(entry)
                st.success("Saved to vault!")
        with col_clear:
            if st.button("Clear Vault"):
                st.session_state["scenario_vault"] = []

        vault = st.session_state.get("scenario_vault", [])
        if not vault:
            st.markdown(
                f"<div style='color: {COLORS['text_muted']};"
                " font-size: 0.9rem; margin-top: 16px;'>"
                "No saved scenarios yet. Run inference and save a"
                " scenario above.</div>",
                unsafe_allow_html=True,
            )
            return

        df_vault = pd.DataFrame(vault)
        display_cols = [
            "label", "final", "RF", "XGB", "SVR",
            "clay", "annual_precip", "peat_frac", "NDVI",
        ]
        display_cols = [c for c in display_cols if c in df_vault.columns]
        df_display = df_vault[display_cols].copy()
        df_display.rename(
            columns={"final": "Stacked (Mg/ha)", "label": "Scenario"},
            inplace=True,
        )

        _fmt_vault = {
            "Stacked (Mg/ha)": "{:.1f}", "RF": "{:.1f}",
            "XGB": "{:.1f}", "SVR": "{:.1f}",
            "clay": "{:.1f}", "annual_precip": "{:.0f}",
            "peat_frac": "{:.2f}", "NDVI": "{:.2f}",
        }
        try:
            _styled_vault = df_display.style.format(_fmt_vault).background_gradient(
                subset=["Stacked (Mg/ha)"], cmap="Greens"
            )
        except ImportError:
            _styled_vault = df_display.style.format(_fmt_vault)
        st.dataframe(
            _styled_vault,
            width='stretch',
            height=min(350, 60 + len(df_vault) * 40),
        )

        if len(vault) > 1:
            st.markdown("")
            labels = [v["label"] for v in vault]
            fig = go.Figure()
            for model in ["RF", "XGB", "SVR", "final"]:
                display = "Stacked" if model == "final" else model
                style = MODEL_STYLES[display]
                vals = [v[model] for v in vault]
                fig.add_trace(go.Bar(
                    name=display, x=labels, y=vals,
                    marker=dict(
                        color=COLORS.get(display, COLORS["Stacked"]),
                        pattern=dict(
                            shape=style["pattern"],
                            fgcolor="rgba(0,0,0,0.15)",
                            size=6,
                        ),
                    ),
                    opacity=0.85 if display != "Stacked" else 1.0,
                ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                font=PLOTLY_FONT,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                barmode="group",
                margin=dict(t=20, b=40, l=10, r=10),
                height=300,
                legend=dict(
                    orientation="h", y=1.12,
                    font=dict(size=10, color=COLORS["text_primary"]),
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(
                    tickfont=dict(
                        size=10, color=COLORS["text_primary"]
                    ),
                ),
                yaxis=dict(
                    title="AGB (Mg/ha)",
                    gridcolor="rgba(0,0,0,0.06)",
                    title_font_color=COLORS["text_primary"],
                    color=COLORS["text_primary"],
                ),
            )
            st.plotly_chart(
                fig, width='stretch',
                config={"displayModeBar": False},
            )

    # ════════════════════════════════════════════════════════
    # PAGE: MODEL STORY
    # ════════════════════════════════════════════════════════
    def _page_model_story(self) -> None:
        st.markdown(
            f"<div style='font-family: Inter, sans-serif;"
            f" font-size: 1.6rem; font-weight: 700;"
            f" color: {COLORS['text_primary']}; margin-bottom: 4px;'>"
            "Ensemble Lineage &amp; Scientific Methodology</div>"
            f"<div style='font-size: 0.78rem;"
            f" color: {COLORS['text_mid']}; margin-bottom: 24px;'>"
            "Stacking Architecture &middot;"
            " Frontiers in Plant Science (2025) &middot;"
            " MPI Principles"
            "</div>",
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3 = st.tabs([
            "ARCHITECTURE",
            "FEATURE IMPORTANCE",
            "PRECIP x CLAY SCIENCE",
        ])

        with tab1:
            self._render_architecture_story()
        with tab2:
            self._render_feature_importance()
        with tab3:
            self._render_precip_clay_science()

    # ── Architecture Story ────────────────────────────────────
    def _render_architecture_story(self) -> None:
        st.markdown(
            "<p class='section-title'>"
            "Stacked Ensemble &mdash; Two-Stage Architecture</p>",
            unsafe_allow_html=True,
        )

        col_text, col_diagram = st.columns([1, 1], gap="large")

        with col_text:
            st.markdown(
                "<div class='tremor-card'>"
                f"<div style='font-size: 1rem; font-weight: 700;"
                f" color: {COLORS['positive']}; margin-bottom: 12px;'>"
                "Level 0 &mdash; Base Learners</div>"
                f"<div style='font-size: 0.9rem;"
                f" color: {COLORS['text_body']}; line-height: 1.65;"
                " margin-bottom: 16px;'>"
                "Three heterogeneous base estimators are trained on the"
                " full feature matrix (9 covariates). Each captures a"
                " different aspect of the biomass-environment"
                " relationship:"
                "<br><br>"
                f"<span style='color: {COLORS['RF']};"
                " font-weight: 600;'>RF</span>"
                " &mdash; Random Forest exploits non-linear interactions"
                " via bagged decision trees; robust to outliers in"
                " clay/peat distributions.<br><br>"
                f"<span style='color: {COLORS['XGB']};"
                " font-weight: 600;'>XGB</span>"
                " &mdash; Gradient-boosted trees model residuals"
                " sequentially, excelling at capturing hydraulic stress"
                " gradients.<br><br>"
                f"<span style='color: {COLORS['SVR']};"
                " font-weight: 600;'>SVR</span>"
                " &mdash; Support Vector Regression with RBF kernel acts"
                " as a smooth global interpolator, anchoring predictions"
                " in low-data regions."
                "</div>"
                f"<div style='font-size: 1rem; font-weight: 700;"
                f" color: {COLORS['Stacked']}; margin-bottom: 12px;'>"
                "Level 1 &mdash; Ridge Meta-Learner</div>"
                f"<div style='font-size: 0.9rem;"
                f" color: {COLORS['text_body']}; line-height: 1.65;'>"
                "Out-of-fold (OOF) predictions from the base models form"
                " a (N, 3) meta-feature matrix. Ridge regression is"
                " fitted on these OOF predictions exclusively &mdash;"
                " preventing target leakage &mdash; and learns an optimal"
                " linear combination. L2 regularisation prevents any"
                " single base model from dominating the consensus."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_diagram:
            # Sankey — fixed arrangement so x/y coords are honoured exactly.
            # Link values are kept uniform (all 1) for feature→model and
            # model→ridge so no node inflates; ridge→AGB uses 3 to match
            # ridge's total inflow and keep both terminal nodes the same size.
            rf_rgba = hex_to_rgba(COLORS["RF"], 0.12)
            xgb_rgba = hex_to_rgba(COLORS["XGB"], 0.12)
            svr_rgba = hex_to_rgba(COLORS["SVR"], 0.12)
            rf_rgba_s = hex_to_rgba(COLORS["RF"], 0.28)
            xgb_rgba_s = hex_to_rgba(COLORS["XGB"], 0.28)
            svr_rgba_s = hex_to_rgba(COLORS["SVR"], 0.28)
            stacked_rgba = hex_to_rgba(COLORS["Stacked"], 0.28)

            fig = go.Figure(go.Sankey(
                arrangement="fixed",
                node=dict(
                    label=[
                        "Clay, Sand, Silt",      # 0
                        "Elevation",              # 1
                        "Precip, Peat, NDVI",     # 2
                        "Hydraulic Stress",        # 3
                        "Precip x Clay",           # 4
                        "Random Forest",           # 5
                        "XGBoost",                 # 6
                        "SVR",                     # 7
                        "Ridge Regression Meta",   # 8
                        "",                        # 9 — label removed, hover retained
                    ],
                    x=[0.01, 0.01, 0.01, 0.01, 0.01,
                       0.38, 0.38, 0.38,
                       0.68, 0.92],
                    y=[0.10, 0.25, 0.44, 0.62, 0.80,
                       0.12, 0.50, 0.88,
                       0.50, 0.50],
                    color=[
                        "rgba(180, 200, 230, 0.85)",       # 0 feature
                        "rgba(180, 200, 230, 0.85)",       # 1 feature
                        "rgba(180, 200, 230, 0.85)",       # 2 feature
                        "rgba(180, 200, 230, 0.85)",       # 3 feature
                        "rgba(180, 200, 230, 0.85)",       # 4 feature
                        hex_to_rgba(COLORS["RF"], 0.18),   # 5 RF
                        hex_to_rgba(COLORS["XGB"], 0.18),  # 6 XGB
                        hex_to_rgba(COLORS["SVR"], 0.18),  # 7 SVR
                        "rgba(120, 185, 155, 0.35)",       # 8 meta-model
                        "rgba(90, 160, 130, 0.95)",        # 9 output
                    ],
                    line=dict(color="#E5E7EB", width=1),
                    pad=35,
                    thickness=18,
                ),
                textfont=dict(
                    family="Arial",
                    size=13,
                    color="#000000",
                ),
                link=dict(
                    source=[
                        0, 0, 0, 1, 1, 1, 2, 2, 2,
                        3, 3, 3, 4, 4, 4,
                        5, 6, 7,
                        8,
                    ],
                    target=[
                        5, 6, 7, 5, 6, 7, 5, 6, 7,
                        5, 6, 7, 5, 6, 7,
                        8, 8, 8,
                        9,
                    ],
                    # All feature→model and model→ridge links are value=1 so
                    # no base model or meta node balloons. Ridge→AGB is 3 to
                    # match Ridge's inflow (3×1) and keep terminal nodes equal.
                    value=[
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1,
                        1, 1, 1,
                        3,
                    ],
                    color=[
                        rf_rgba, xgb_rgba, svr_rgba,
                        rf_rgba, xgb_rgba, svr_rgba,
                        rf_rgba, xgb_rgba, svr_rgba,
                        rf_rgba, xgb_rgba, svr_rgba,
                        rf_rgba, xgb_rgba, svr_rgba,
                        rf_rgba_s, xgb_rgba_s, svr_rgba_s,
                        stacked_rgba,
                    ],
                ),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="Arial",
                    size=13,
                    color="#111827",
                ),
                margin=dict(l=20, r=20, t=30, b=20),
                height=650,
            )
            st.plotly_chart(
                fig, width='stretch',
                config={"displayModeBar": False},
            )

        # Model metrics table
        st.markdown("")
        st.markdown(
            "<p class='section-title'>"
            "Published Performance Metrics</p>",
            unsafe_allow_html=True,
        )
        metrics_data = {
            "Model": [
                "Random Forest", "XGBoost", "SVR",
                "Stacked Ensemble",
            ],
            "R2": [0.872, 0.884, 0.831, 0.913],
            "RMSE (Mg/ha)": [28.4, 26.9, 33.7, 22.1],
            "MAE (Mg/ha)": [19.2, 18.4, 24.1, 15.8],
        }
        _fmt_metrics = {"R2": "{:.3f}", "RMSE (Mg/ha)": "{:.1f}", "MAE (Mg/ha)": "{:.1f}"}
        try:
            _styled_metrics = (
                pd.DataFrame(metrics_data)
                .style.format(_fmt_metrics)
                .background_gradient(subset=["R2"], cmap="Greens")
            )
        except ImportError:
            _styled_metrics = pd.DataFrame(metrics_data).style.format(_fmt_metrics)
        st.dataframe(
            _styled_metrics,
            width='stretch',
            hide_index=True,
        )

    # ── Feature Importance ────────────────────────────────────
    def _render_feature_importance(self) -> None:
        st.markdown(
            "<p class='section-title'>"
            "Aggregated Feature Importance &mdash;"
            " Random Forest Proxy</p>",
            unsafe_allow_html=True,
        )

        features = [
            "NDVI", "peat_frac", "precip_clay_interaction",
            "annual_precip", "Hydraulic_Stress", "clay",
            "elev", "sand", "silt",
        ]
        importances = [
            0.234, 0.198, 0.162, 0.128, 0.094,
            0.078, 0.054, 0.032, 0.020,
        ]
        colours = [
            COLORS["positive"]
            if f in ["NDVI", "peat_frac", "precip_clay_interaction"]
            else COLORS["warning"]
            if f in ["annual_precip", "Hydraulic_Stress"]
            else COLORS["text_muted"]
            for f in features
        ]

        df_imp = pd.DataFrame({
            "Feature": features,
            "Importance": importances,
            "Color": colours,
        })
        df_imp = df_imp.sort_values("Importance")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Feature"],
            orientation="h",
            marker=dict(
                color=df_imp["Color"].tolist(),
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=[f"{v:.3f}" for v in df_imp["Importance"]],
            textposition="outside",
            textfont=dict(
                size=11, color=COLORS["text_primary"], family="Inter"
            ),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            font=PLOTLY_FONT,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=10, r=60),
            height=360,
            xaxis=dict(
                title="Importance",
                gridcolor="rgba(0,0,0,0.06)",
                title_font_color=COLORS["text_primary"],
                color=COLORS["text_primary"],
            ),
            yaxis=dict(
                tickfont=dict(
                    size=12, color=COLORS["text_primary"],
                    family="Inter",
                ),
            ),
        )
        st.plotly_chart(
            fig, width='stretch',
            config={"displayModeBar": False},
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                "<div class='tremor-card' style='border-left: 3px solid"
                f" {COLORS['RF']};'>"
                f"<div style='font-size: 0.68rem;"
                f" color: {COLORS['text_muted']};"
                " text-transform: uppercase; margin-bottom: 8px;'>"
                "Vegetation Signal</div>"
                f"<div style='font-size: 0.88rem;"
                f" color: {COLORS['text_body']}; line-height: 1.55;'>"
                "NDVI (23.4%) dominates &mdash; canopy greenness is a"
                " direct proxy for foliar biomass density in dense"
                " peat-swamp formations."
                "</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                "<div class='tremor-card' style='border-left: 3px solid"
                f" {COLORS['SVR']};'>"
                f"<div style='font-size: 0.68rem;"
                f" color: {COLORS['text_muted']};"
                " text-transform: uppercase; margin-bottom: 8px;'>"
                "Peat Substrate</div>"
                f"<div style='font-size: 0.88rem;"
                f" color: {COLORS['text_body']}; line-height: 1.55;'>"
                "Peat fraction (19.8%) encodes soil carbon depth and"
                " moisture retention &mdash; both critical for AGB"
                " accumulation on Bornean domes."
                "</div></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                "<div class='tremor-card' style='border-left: 3px solid"
                f" {COLORS['XGB']};'>"
                f"<div style='font-size: 0.68rem;"
                f" color: {COLORS['text_muted']};"
                " text-transform: uppercase; margin-bottom: 8px;'>"
                "Interaction Term</div>"
                f"<div style='font-size: 0.88rem;"
                f" color: {COLORS['text_body']}; line-height: 1.55;'>"
                "precip x clay (16.2%) &mdash; a derived waterlogging"
                " proxy that outranks raw precipitation alone in"
                " importance rankings."
                "</div></div>",
                unsafe_allow_html=True,
            )

    # ── Precip x Clay Science (Futuristic Dark Module) ────────
    def _render_precip_clay_science(self) -> None:
        st.markdown(
            "<p class='section-title'>"
            "The Precip &times; Clay Interaction Term &mdash;"
            " Scientific Basis</p>",
            unsafe_allow_html=True,
        )

        col_theory, col_chart = st.columns([1, 1], gap="large")

        with col_theory:
            st.markdown(
                "<div class='dark-module'>"
                "<div style='font-size: 0.95rem; font-weight: 700;"
                " color: #10B981; margin-bottom: 12px;'>"
                "Why Does This Interaction Matter?</div>"
                "<div style='font-size: 0.9rem;"
                " color: #D1D5DB; line-height: 1.7;"
                " margin-bottom: 14px;'>"
                "In tropical peat-swamp forests, precipitation alone"
                " is a poor predictor of AGB. High rainfall in"
                " <em style='color: #6EE7B7;'>well-drained sandy soils</em>"
                " rapidly leaches nutrients and dries out, suppressing"
                " biomass. The same rainfall volume in"
                " <em style='color: #6EE7B7;'>clay-rich soils</em>"
                " produces persistent waterlogging &mdash; raising the"
                " water table and dramatically altering decomposition"
                " rates and root architecture."
                "</div>"
                "<div style='font-size: 0.9rem; font-weight: 700;"
                " color: #10B981; margin-bottom: 8px;'>Formula</div>"
                "<div style='font-family: monospace; font-size: 0.9rem;"
                " background: #1F2937; padding: 10px 14px;"
                " border-radius: 8px; border-left: 3px solid #10B981;"
                " color: #6EE7B7; margin-bottom: 14px;'>"
                "precip_clay_interaction = annual_precip &times; clay"
                "</div>"
                "<div style='font-size: 0.9rem;"
                " color: #D1D5DB; line-height: 1.7;'>"
                "<strong style='color: #F9FAFB;'>"
                "Regime classification:</strong><br>"
                "<span style='color: #10B981;'>"
                "&#10003; Low  (&lt;20,000)</span>"
                " &mdash; Arid or sandy; minimal waterlogging<br>"
                "<span style='color: #F59E0B;'>"
                "&#9888; Moderate (20k-100k)</span>"
                " &mdash; Transitional; partial peat inundation<br>"
                "<span style='color: #EF4444;'>"
                "&#9888; High  (&gt;100,000)</span>"
                " &mdash; Deep peat domes; anaerobic conditions"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_chart:
            clay_vals = np.linspace(5, 65, 40)
            precip_vals = np.linspace(800, 4500, 40)
            clay_grid, precip_grid = np.meshgrid(
                clay_vals, precip_vals
            )
            agb_surface = (
                80
                + 0.5 * clay_grid
                + 0.03 * precip_grid
                + 0.0001 * (precip_grid * clay_grid)
            )

            fig = go.Figure(data=[go.Surface(
                x=clay_vals, y=precip_vals, z=agb_surface,
                colorscale=EMERALD_SLATE_SCALE,
                opacity=0.90,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="AGB (Mg/ha)",
                        font=dict(color="#D1D5DB", size=10),
                    ),
                    tickfont=dict(color="#D1D5DB", size=9),
                    len=0.7,
                ),
                contours=dict(
                    z=dict(
                        show=True, start=80, end=280, size=20,
                        color="rgba(16,185,129,0.15)", width=1,
                    ),
                ),
            )])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title="Clay (%)",
                        color="#9CA3AF",
                        backgroundcolor="#111827",
                        gridcolor="rgba(16,185,129,0.12)",
                    ),
                    yaxis=dict(
                        title="Precip (mm/yr)",
                        color="#9CA3AF",
                        backgroundcolor="#111827",
                        gridcolor="rgba(16,185,129,0.12)",
                    ),
                    zaxis=dict(
                        title="AGB (Mg/ha)",
                        color="#9CA3AF",
                        backgroundcolor="#111827",
                        gridcolor="rgba(16,185,129,0.12)",
                    ),
                    bgcolor="#0A0A0A",
                ),
                paper_bgcolor="#0A0A0A",
                margin=dict(t=10, b=10, l=0, r=0),
                height=400,
                font=dict(
                    family="Inter, sans-serif",
                    color="#D1D5DB",
                ),
            )
            st.plotly_chart(
                fig, width='stretch',
                config={"displayModeBar": False},
            )

        # Literature citation block — dark themed
        st.markdown("")
        st.markdown(
            "<div class='dark-card' style='border-left: 3px solid"
            " #10B981;'>"
            "<div style='font-size: 0.84rem; font-weight: 700;"
            " color: #10B981; margin-bottom: 8px;'>"
            "Scientific Literature Context</div>"
            "<div style='font-size: 0.88rem;"
            " color: #9CA3AF; line-height: 1.65;'>"
            "This interaction approach aligns with"
            " <em style='color: #D1D5DB;'>Hastie et al. (2009)</em>"
            " &mdash; multiplicative interaction terms are essential"
            " when two covariates jointly control a non-linear outcome."
            " In the context of tropical peat forests,"
            " <em style='color: #D1D5DB;'>Page et al. (2011)</em>"
            " documented that waterlogging-driven anaerobic decomposition"
            " is a primary determinant of peat depth and therefore AGB"
            " standing stock. The"
            " <em style='color: #D1D5DB;'>MPI Biomass Modelling"
            " Framework</em> similarly recommends soil-hydrology"
            " interaction features for any AGB model operating in"
            " inundation-prone biomes."
            "</div></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    app = PremiumBiomassApp()
    app.run()


if __name__ == "__main__":
    main()
