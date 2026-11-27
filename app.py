import os
# Ensure YOLO config directory is writable across environments (local Windows sandbox, Linux container)
if not os.environ.get("YOLO_CONFIG_DIR"):
    _fallback_yolo = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yolo")
    os.makedirs(_fallback_yolo, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = _fallback_yolo

import streamlit as st
import streamlit.components.v1 as components
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import edge_tts
import asyncio
import hashlib
import time
import threading
import urllib.request
import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv

# ── RESEND EMAIL ALERT SERVICE ──
from alerts.email_alerts import EmailAlerter

# ── SECURITY & VALIDATION UTILITIES ──
from utils.security import (
    validate_stream_url,
    normalize_camera_url,
    verify_model_hash,
    discover_paired_camera,
    scan_network_cameras,
    discover_cameras_on_lan,
    verify_camera_handshake,
    resolve_paired_camera,
    load_paired_cameras,
    save_camera_slot,
    get_paired_camera,
    clean_ip_address,
    get_local_ip,
    get_all_local_ips,
    DEFAULT_BEST_PT_SHA256,
)

# Load environment variables from .env
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# MODULE INSTANCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_email_alerter():
    """Module-level single EmailAlerter worker cached across Streamlit reruns."""
    return EmailAlerter()

email_alerter = get_email_alerter()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FIRE_CONF_THRESHOLD   = 0.80
CROWD_THRESHOLD       = 1
FRAME_SKIP            = 4
EMAIL_COOLDOWN_SEC    = 120
AUDIO_COOLDOWN_SEC    = 120
ALERT_CLEAR_GRACE_SEC = 3.0  # Debounce: delay clearing alerts by 3s to prevent visual banner flicker
SILENCE_DURATION_SEC  = 60   # Cooldown: mute alerts for 60s when operator clicks Silence

# Standard Hindi Voice Alert Prompts
FIRE_TTS_TEXT  = "कृपया ध्यान दें। मेन गेट पर आग डिटेक्ट हुई है। तुरंत फायर सेफ्टी टीम डिस्पैच करें।"
CROWD_TTS_TEXT = "कृपया ध्यान दें। मेन गेट एंट्रेंस पर ओवरक्राउडिंग डिटेक्ट हुई है। तुरंत सिक्योरिटी स्टाफ डिस्पैच करें।"

# Thread-safety lock for YOLO / PyTorch model inference across threads
model_inference_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HazardLens | Enterprise NVR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Minimalist Enterprise NVR Design System (Linear / Verkada / Vercel)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:               #09090B;
  --surface:          #111114;
  --surface-elevated: #18181B;
  --viewport:         #0C0C0E;
  --border:           #27272A;
  --border-hover:     #3F3F46;
  --border-subtle:    #1F1F23;
  --text:             #FAFAFA;
  --text-muted:       #71717A;
  --text-sub:         #A1A1AA;
  --emerald:          #10B981;
  --emerald-dim:      rgba(16, 185, 129, 0.1);
  --emerald-border:   rgba(16, 185, 129, 0.25);
  --red:              #EF4444;
  --red-dim:          rgba(239, 68, 68, 0.1);
  --red-border:       rgba(239, 68, 68, 0.25);
  --amber:            #F59E0B;
  --amber-dim:        rgba(245, 158, 11, 0.1);
  --amber-border:     rgba(245, 158, 11, 0.25);
  --sans:             'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --mono:             'JetBrains Mono', ui-monospace, monospace;
}

/* ── BASE APPLICATION ── */
.stApp, .main {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
}
/* Keep header transparent so sidebar toggle button is visible at top-left */
header[data-testid="stHeader"] {
  background: transparent !important;
  color: var(--text) !important;
  pointer-events: none !important;
  z-index: 999999 !important;
}
/* Allow clicks on the toolbar and expand controls */
[data-testid="stToolbar"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  visibility: visible !important;
  display: flex !important;
  background: transparent !important;
  pointer-events: auto !important;
}
/* Hide ONLY the right-side toolbar items, deploy button, status widget, footer, and decorations */
[data-testid="stToolbarActions"],
[data-testid="stDeployButton"],
[data-testid="stStatusWidget"],
#MainMenu,
footer,
[data-testid="stDecoration"] {
  visibility: hidden !important;
  display: none !important;
}
/* Ensure sidebar reopen/expand button remains prominently visible and clickable */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button,
button[data-testid="stExpandSidebarButton"],
button[data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  color: #FAFAFA !important;
  background: #18181B !important;
  border: 1px solid #27272A !important;
  border-radius: 6px !important;
  padding: 6px 10px !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  cursor: pointer !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="collapsedControl"]:hover,
[data-testid="stExpandSidebarButton"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {
  background: #27272A !important;
  border-color: var(--border-hover) !important;
  color: #FFFFFF !important;
}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
  fill: #FAFAFA !important;
  color: #FAFAFA !important;
  stroke: #FAFAFA !important;
}
*, *::before, *::after {
  box-sizing: border-box;
}
div.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 2rem !important;
  max-width: 100% !important;
}

/* ── SIDEBAR & TOP ALIGNMENT ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarHeader"] {
  padding: 0.5rem 1rem 0 1rem !important;
  min-height: auto !important;
  height: auto !important;
}
[data-testid="stSidebarHeader"] button {
  color: var(--text-muted) !important;
}
[data-testid="stSidebarHeader"] button:hover {
  color: var(--text) !important;
}
[data-testid="stSidebarUserContent"] {
  padding-top: 0.25rem !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}

/* ── BRANDING ── */
.sb-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-bottom: 14px;
  margin: 0 0 18px 0;
  border-bottom: 1px solid #202024;
}
.sb-badge {
  width: 38px;
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: linear-gradient(180deg, #1F1F23 0%, #141417 100%);
  border: 1px solid #2E2E34;
  border-radius: 10px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.sb-brand-text {
  display: flex;
  flex-direction: column;
}
.sb-title-row {
  display: flex;
  align-items: center;
  gap: 6px;
  line-height: 1;
}
.sb-title {
  font-family: var(--sans);
  font-size: 18px;
  letter-spacing: -0.03em;
  line-height: 1;
}
.sb-title .word-hazard {
  color: #FAFAFA;
  font-weight: 700;
}
.sb-title .word-lens {
  color: #A1A1AA;
  font-weight: 500;
}
.sb-nvr-badge {
  font-family: var(--mono);
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  color: #A1A1AA;
  background: #18181B;
  border: 1px solid #27272A;
  padding: 2px 6px;
  border-radius: 4px;
  letter-spacing: 0.04em;
  line-height: 1;
}
.sb-subtitle {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #71717A;
  margin-top: 4px;
  line-height: 1;
}
.sb-divider {
  height: 1px;
  background: var(--border);
  margin: 20px 0 16px;
}
.sb-section-title {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.sb-settings-grid {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.sb-setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 10px;
  background: var(--surface-elevated);
  border: 1px solid var(--border);
  border-radius: 6px;
}
.sb-setting-label {
  font-family: var(--sans);
  font-size: 11px;
  color: var(--text-muted);
}
.sb-setting-val {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--text);
}

/* ── STREAMLIT INPUTS ── */
div[data-testid="stTextInput"] label {
  font-family: var(--sans) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  margin-bottom: 6px !important;
}
div[data-testid="stTextInput"] input {
  background: var(--surface-elevated) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  border-radius: 6px !important;
  padding: 8px 12px !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--border-hover) !important;
  box-shadow: 0 0 0 1px #3F3F46 !important;
}
div[data-testid="stTextInput"] input::placeholder {
  color: #52525B !important;
}

/* ── BUTTONS ── */
div[data-testid="stButton"] > button {
  font-family: var(--sans) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  border-radius: 6px !important;
  padding: 9px 14px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface-elevated) !important;
  color: var(--text) !important;
  transition: all 0.15s ease !important;
}
div[data-testid="stButton"] > button:hover {
  background: #27272A !important;
  border-color: var(--border-hover) !important;
  color: #FFFFFF !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background: #FAFAFA !important;
  color: #09090B !important;
  border: 1px solid #FAFAFA !important;
  font-weight: 700 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: #E4E4E7 !important;
  border-color: #E4E4E7 !important;
}
button:has(p:contains("DISARM SYSTEM")),
button:has(div:contains("DISARM SYSTEM")) {
  background: var(--red-dim) !important;
  border: 1px solid var(--red-border) !important;
  color: var(--red) !important;
}
button:has(p:contains("DISARM SYSTEM")):hover,
button:has(div:contains("DISARM SYSTEM")):hover {
  background: rgba(239, 68, 68, 0.18) !important;
  border-color: var(--red) !important;
}
button:has(p:contains("Silence Alert")),
button:has(div:contains("Silence Alert")) {
  background: var(--red-dim) !important;
  border: 1px solid var(--red-border) !important;
  color: var(--red) !important;
  margin-top: 6px !important;
}
button:has(p:contains("Silence Alert")):hover,
button:has(div:contains("Silence Alert")):hover {
  background: rgba(239, 68, 68, 0.2) !important;
}

/* ── SLIM HEADER BAR ── */
.hz-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  padding: 10px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 16px;
}
.hz-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.hz-title {
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}
.hz-loc-tag {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-muted);
  padding: 2px 8px;
  background: var(--surface-elevated);
  border: 1px solid var(--border);
  border-radius: 4px;
}
.hz-header-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: var(--surface-elevated);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 12px;
}
.pill-label {
  font-family: var(--sans);
  color: var(--text-muted);
  font-size: 11px;
}
.pill-val {
  font-family: var(--mono);
  font-weight: 600;
  color: var(--text);
  font-size: 11px;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 6px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.status-armed {
  background: var(--red-dim);
  border: 1px solid var(--red-border);
  color: var(--red);
}
.status-standby {
  background: var(--amber-dim);
  border: 1px solid var(--amber-border);
  color: var(--amber);
}

/* ── PULSE DOTS ── */
@keyframes dotPulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.9); }
}
.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
.dot-emerald { background: var(--emerald); animation: dotPulse 1.8s ease-in-out infinite; }
.dot-red     { background: var(--red);     animation: dotPulse 1.0s ease-in-out infinite; }
.dot-amber   { background: var(--amber);   animation: dotPulse 2.0s ease-in-out infinite; }
.dot-muted   { background: #52525B; }

/* ── CAMERA CARDS ── */
.cam-card-hdr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-bottom: none;
  border-radius: 8px 8px 0 0;
}
.cam-card-meta {
  display: flex;
  align-items: center;
  gap: 10px;
}
.cam-id {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--text);
  background: var(--surface-elevated);
  border: 1px solid var(--border);
  padding: 2px 6px;
  border-radius: 4px;
  letter-spacing: 0.02em;
}
.cam-name {
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 500;
  color: var(--text-sub);
}
.cam-card-right {
  display: flex;
  align-items: center;
  gap: 12px;
}
.cam-crowd-count {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  color: var(--text-muted);
}
.badge-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 2px 8px;
  border-radius: 4px;
}
.badge-live {
  background: var(--emerald-dim);
  border: 1px solid var(--emerald-border);
  color: var(--emerald);
}
.badge-warn {
  background: var(--amber-dim);
  border: 1px solid var(--amber-border);
  color: var(--amber);
}
.badge-offline {
  background: var(--surface-elevated);
  border: 1px solid var(--border);
  color: var(--text-muted);
}

/* ── CAMERA VIEWPORT IMAGES ── */
div[data-testid="stImage"] {
  margin: 0 !important;
  padding: 0 !important;
}
div[data-testid="stImage"] img {
  display: block;
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 0 0 8px 8px;
  background: var(--viewport);
}
/* Hide Streamlit default image fullscreen hover button and element toolbar */
.stElementToolbar,
[data-testid="stElementToolbar"],
[data-testid="stElementToolbarButton"],
[data-testid="stElementToolbarButtonContainer"],
[data-testid="StyledFullScreenButton"],
button[title*="fullscreen" i],
button[aria-label*="fullscreen" i],
button[title="View fullscreen"],
button[title="Exit fullscreen"],
div[data-testid="stImage"] button,
div[data-testid="stImage"] .stElementToolbar,
div[data-testid="stImage"] [data-testid="stElementToolbar"],
div[data-testid="stImage"] [data-testid="stElementToolbarButtonContainer"],
div[data-testid="stFullScreenFrame"] [data-testid="stElementToolbar"],
.stApp [data-testid="stElementToolbar"],
.stApp .stElementToolbar {
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
  height: 0 !important;
  width: 0 !important;
  max-height: 0 !important;
  max-width: 0 !important;
  overflow: hidden !important;
}

/* ── ALERT BANNER ── */
@keyframes si {
  from { transform: translateY(-4px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
.a-banner {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 18px;
  margin-bottom: 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  animation: si 0.2s ease;
}
.a-fire {
  border-left: 3px solid var(--red);
}
.a-crwd {
  border-left: 3px solid var(--amber);
}
.a-icon {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 6px;
}
.a-fire .a-icon {
  background: var(--red-dim);
  color: var(--red);
}
.a-crwd .a-icon {
  background: var(--amber-dim);
  color: var(--amber);
}
.a-content {
  flex: 1;
}
.a-kicker {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 2px;
}
.a-fire .a-kicker { color: var(--red); }
.a-crwd .a-kicker { color: var(--amber); }
.a-ttl {
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
}
.a-msg {
  font-family: var(--sans);
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 2px;
}

/* ── FOOTER STATUS BAR ── */
.ftr {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 8px 20px;
  padding: 10px 16px;
  margin-top: 24px;
  border: 1px solid var(--border);
  background: var(--surface);
  border-radius: 8px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-muted);
}

/* ── TOAST & SCROLLBAR ── */
div[data-testid="stToast"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
  border-radius: 8px !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hover); border-radius: 2px; }

/* Hidden background audio player */
div[data-testid="stAudio"] {
  display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
if 'system_armed'      not in st.session_state: st.session_state.system_armed      = False
if 'is_operator'       not in st.session_state: st.session_state.is_operator       = True
if 'paired_camera_url' not in st.session_state: st.session_state.paired_camera_url  = None
if 'do_clear_alert'    not in st.session_state: st.session_state.do_clear_alert    = False
if 'active_alert'      not in st.session_state: st.session_state.active_alert      = None
if 'hazard_clear_ts'   not in st.session_state: st.session_state.hazard_clear_ts   = None
if 'silenced_until'    not in st.session_state: st.session_state.silenced_until    = 0.0
if 'last_audio_ts'     not in st.session_state: st.session_state.last_audio_ts     = {"FIRE": 0.0, "CROWD": 0.0}
if 'camera_worker'     not in st.session_state: st.session_state.camera_worker     = None
if 'live_crowd_count'           not in st.session_state: st.session_state.live_crowd_count           = 0
if 'active_audio_b64'           not in st.session_state: st.session_state.active_audio_b64           = None
if 'active_audio_id'            not in st.session_state: st.session_state.active_audio_id            = 0
if 'audio_trigger_id'           not in st.session_state: st.session_state.audio_trigger_id           = 0
if 'stop_audio_id'              not in st.session_state: st.session_state.stop_audio_id              = 0
if '_last_rendered_audio_state' not in st.session_state: st.session_state._last_rendered_audio_state = None
if 'audio_expire_ts'            not in st.session_state: st.session_state.audio_expire_ts            = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached for zero-lag reloads with SHA-256 integrity check)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_twin_models():
    # Model integrity check to mitigate pickle deserialization risks (RCE)
    ok, val = verify_model_hash('best.pt')
    if not ok:
        st.error(f"CRITICAL SECURITY ALERT: Model integrity verification failed: {val}")
        st.stop()
    fire_mod = YOLO('best.pt')      # Custom Fire / Smoke model
    base_mod = YOLO('yolov8n.pt')   # Standard model for crowd counting
    return fire_mod, base_mod

fire_model, base_model = load_twin_models()

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def lock_operator():
    pass

def dummy_silence(cam_name):
    st.session_state.dummy_silence_cam = cam_name

def silence_alerts():
    now = time.time()
    st.session_state.silenced_until   = now + SILENCE_DURATION_SEC
    st.session_state.do_clear_alert   = True
    st.session_state.active_alert     = None
    st.session_state.hazard_clear_ts  = None
    st.session_state.active_audio_b64 = None
    st.session_state.audio_expire_ts  = 0.0
    st.session_state.stop_audio_id    = st.session_state.get('stop_audio_id', 0) + 1
    st.session_state.active_audio_id  += 1
    if "last_audio_ts" in st.session_state and isinstance(st.session_state.last_audio_ts, dict):
        st.session_state.last_audio_ts["FIRE"] = 0.0
        st.session_state.last_audio_ts["CROWD"] = 0.0
    st.session_state.silence_toast_pending = True

def disarm_system():
    st.session_state.system_armed     = False
    st.session_state.do_clear_alert   = True
    st.session_state.active_alert     = None
    st.session_state.hazard_clear_ts  = None
    st.session_state.silenced_until   = 0.0
    st.session_state.active_audio_b64 = None
    st.session_state.audio_expire_ts  = 0.0
    st.session_state.stop_audio_id    = st.session_state.get('stop_audio_id', 0) + 1
    st.session_state.active_audio_id  += 1
    if "last_audio_ts" in st.session_state and isinstance(st.session_state.last_audio_ts, dict):
        st.session_state.last_audio_ts["FIRE"] = 0.0
        st.session_state.last_audio_ts["CROWD"] = 0.0
    if st.session_state.get('camera_worker') is not None:
        st.session_state.camera_worker.stop()
        st.session_state.camera_worker = None

def arm_system():
    st.session_state.system_armed = True
    # 1-Click Auto-Link & Auto-Heal: proactively discover camera if unconfigured or offline
    try:
        resolved = resolve_paired_camera("CAM 01")
        if resolved:
            st.session_state.paired_camera_url = resolved
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def create_placeholder(cam_name, status="STANDBY"):
    """Clean, matte 16:9 box with subtle centered status text."""
    h, w = 360, 640
    # Matte dark surface #0C0C0E (BGR: 14, 12, 12)
    img = np.full((h, w, 3), (14, 12, 12), dtype=np.uint8)

    # Subtle centered text in muted zinc #71717A (BGR: 122, 113, 113)
    cx, cy = w // 2, h // 2
    text = f"{cam_name} · {status}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2),
                font, scale, (122, 113, 113), thickness, cv2.LINE_AA)
    return img

# ---------------------------------------------------------
# --- EDGE-TTS HINDI NEURAL VOICE (SAFE ISOLATED THREAD) ---
# ---------------------------------------------------------
def _generate_tts_sync(text, filename):
    """Run edge-tts async generation safely inside an isolated thread with a clean event loop."""
    def _thread_target():
        try:
            async def _generate():
                communicate = edge_tts.Communicate(text, "hi-IN-SwaraNeural")
                await communicate.save(filename)
            asyncio.run(_generate())
        except Exception as e:
            print(f"Edge-TTS thread generation error: {e}")

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join(timeout=5.0)

@st.cache_resource
def prewarm_tts_assets():
    """Ensures both Hindi voice alert MP3s exist on disk before monitoring starts."""
    if not os.path.exists("crowd_alert.mp3") or os.path.getsize("crowd_alert.mp3") == 0:
        _generate_tts_sync(CROWD_TTS_TEXT, "crowd_alert.mp3")
    if not os.path.exists("fire_alert.mp3") or os.path.getsize("fire_alert.mp3") == 0:
        _generate_tts_sync(FIRE_TTS_TEXT, "fire_alert.mp3")
    return True

prewarm_tts_assets()

def trigger_hindi_alert(text):
    """Sets active audio in session state for persistent playback across fragment reruns."""
    try:
        if "ओवरक्राउडिंग" in text or text == CROWD_TTS_TEXT:
            target_file = "crowd_alert.mp3"
        else:
            target_file = "fire_alert.mp3"

        if not os.path.exists(target_file) or os.path.getsize(target_file) == 0:
            _generate_tts_sync(text, target_file)

        if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
            with open(target_file, "rb") as f:
                audio_bytes = f.read()
            st.session_state.active_audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
            st.session_state.active_audio_id += 1
            st.session_state.audio_trigger_id = st.session_state.get('audio_trigger_id', 0) + 1
            st.session_state.audio_expire_ts = time.time() + 10.0
    except Exception as e:
        print(f"TTS Error: {e}")

# ---------------------------------------------------------
# --- STREAMLIT HTML / AUDIO CONTROLLER HELPER ---
# ---------------------------------------------------------
try:
    from streamlit import deprecation_util as _deprec_util  # type: ignore
    class _SuppressHtmlDeprecation(object):
        def filter(self, record):
            return "components.v1.html" not in record.getMessage()
    _deprec_util._LOGGER.addFilter(_SuppressHtmlDeprecation())
except Exception:
    pass

def _render_audio_html(html_snippet: str):
    """
    Renders the HTML/JS Web Audio snippet cleanly with zero layout shift
    and suppresses deprecation warnings.
    """
    if hasattr(st, "iframe"):
        try:
            st.iframe(html_snippet, height=0)
            return
        except Exception:
            pass
    # Fallback to components.html with height=0, width=0
    components.html(html_snippet, height=0, width=0)

def play_persistent_audio(b64_audio: str | None = None, audio_id: int = 0, force: bool = False):
    """
    Plays audio via the top-level parent window Web Audio controller to prevent
    fragment unmount and Diff resets every 0.5s. Loops infinitely with a 1.2s pause
    until explicitly stopped.
    Only renders HTML/JS snippet when audio state actually transitions!
    """
    if b64_audio:
        trigger_id = st.session_state.get("audio_trigger_id", audio_id)
        current_state = f"PLAY:{audio_id}:{trigger_id}"
    else:
        stop_id = st.session_state.get("stop_audio_id", 0)
        current_state = f"STOP:{stop_id}"

    last_state = st.session_state.get("_last_rendered_audio_state")
    if not force and last_state == current_state:
        # Audio state unchanged — do NOT call HTML renderer on periodic 0.5s fragment ticks!
        return

    st.session_state["_last_rendered_audio_state"] = current_state

    try:
        if not b64_audio:
            stop_script = """
            <script>
            (function() {
              const w = window.parent;
              if (w) {
                if (w._hzLoopTimer) {
                  try { clearTimeout(w._hzLoopTimer); } catch(e) {}
                  w._hzLoopTimer = null;
                }
                if (w._hzAudio) {
                  try {
                    w._hzAudio.loop = false;
                    w._hzAudio.pause();
                    w._hzAudio.currentTime = 0;
                  } catch(e) {}
                  w._hzAudio = null;
                }
                w._hzLastAudioId = null;
              }
            })();
            </script>
            """
            _render_audio_html(stop_script)
            return

        script = f"""
        <script>
        (function() {{
          const w = window.parent;
          const audioId = {audio_id};
          if (w._hzLastAudioId !== audioId) {{
            w._hzLastAudioId = audioId;
            if (w._hzLoopTimer) {{
              try {{ clearTimeout(w._hzLoopTimer); }} catch(e) {{}}
              w._hzLoopTimer = null;
            }}
            if (w._hzAudio) {{
              try {{ w._hzAudio.pause(); }} catch(e) {{}}
            }}
            w._hzAudio = new w.Audio("data:audio/mp3;base64,{b64_audio}");
            w._hzAudio.loop = true;
            w._hzAudio.onended = function() {{
              if (w._hzAudio) {{
                w._hzLoopTimer = setTimeout(function() {{
                  if (w._hzAudio) {{
                    w._hzAudio.currentTime = 0;
                    w._hzAudio.play().catch(function(e){{}});
                  }}
                }}, 1200);
              }}
            }};
            w._hzAudio.play().catch(e => console.warn("Audio play blocked:", e));
          }}
        }})();
        </script>
        """
        _render_audio_html(script)
    except Exception as e:
        print(f"Parent-window audio controller error: {e}")

# ---------------------------------------------------------
# --- DECOUPLED ASYNC DUAL-YOLO INFERENCE WORKER ---
# ---------------------------------------------------------
class InferenceWorker:
    """
    Decoupled background worker for async dual-YOLO inference.
    Runs fire_model (best.pt) and base_model (yolov8n.pt) at imgsz=320 without blocking
    the video capture thread or the Streamlit display rendering loop.
    """
    def __init__(self, fire_mod, base_mod, fire_conf: float = 0.80, crowd_thresh: int = 1):
        self.fire_model = fire_mod
        self.base_model = base_mod
        self.fire_conf = fire_conf
        self.crowd_threshold = crowd_thresh

        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        self.person_count: int = 0
        self.fire_spotted: bool = False
        self.inference_ms: float = 0.0
        self.detections: list[dict] = []
        self._last_processed_seq: int = -1

    def start(self, get_frame_fn):
        if self.running and self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, args=(get_frame_fn,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive() and self.thread != threading.current_thread():
            self.thread.join(timeout=1.0)
        self.thread = None

    def _run_loop(self, get_frame_fn):
        while self.running:
            # Grab latest raw frame from CameraWorker
            frame, seq = get_frame_fn()
            if frame is None or seq == self._last_processed_seq:
                time.sleep(0.005)
                continue

            self._last_processed_seq = seq
            t0 = time.time()
            try:
                with model_inference_lock:
                    # Engine 1: Fire & Smoke (imgsz=320 for 3-4x faster inference)
                    fire_res = self.fire_model(frame, conf=self.fire_conf, imgsz=320, verbose=False)
                    # Engine 2: Base crowd model (imgsz=320 for fast person detection)
                    base_res = self.base_model(frame, imgsz=320, verbose=False)

                inf_time_ms = (time.time() - t0) * 1000.0

                parsed_dets: list[dict] = []
                fire_found = False
                p_count = 0

                # Parse Fire / Smoke detections
                if fire_res and len(fire_res) > 0 and fire_res[0].boxes is not None:
                    for b in fire_res[0].boxes:
                        cls_idx = int(b.cls[0])
                        cls_name = self.fire_model.names.get(cls_idx, "fire")
                        conf = float(b.conf[0])
                        coords = tuple(map(int, b.xyxy[0]))
                        if cls_name.lower() in ("fire", "smoke", "火", "烟"):
                            fire_found = True
                            parsed_dets.append({
                                "bbox": coords,
                                "label": cls_name.upper(),
                                "conf": conf,
                                "type": "fire"
                            })

                # Parse Person detections
                if base_res and len(base_res) > 0 and base_res[0].boxes is not None:
                    for b in base_res[0].boxes:
                        cls_idx = int(b.cls[0])
                        cls_name = self.base_model.names.get(cls_idx, "")
                        if cls_name == "person":
                            p_count += 1
                            conf = float(b.conf[0])
                            coords = tuple(map(int, b.xyxy[0]))
                            parsed_dets.append({
                                "bbox": coords,
                                "label": "PERSON",
                                "conf": conf,
                                "type": "person"
                            })

                with self.lock:
                    self.detections = parsed_dets
                    self.person_count = p_count
                    self.fire_spotted = fire_found
                    self.inference_ms = inf_time_ms

            except Exception:
                pass


# ---------------------------------------------------------
# --- ZERO-BUFFER PERSISTENT STREAM GRABBER (CAMERA WORKER) ---
# ---------------------------------------------------------
class CameraWorker:
    """
    High-performance zero-buffer camera stream grabber with decoupled async YOLO inference.
    - Persistent requests.Session() eliminates TCP+TLS handshakes per frame over Cloudflare Quick Tunnel.
    - Primary mode: Streams MJPEG from {base_url}/video with boundary parser and zero-buffer latest-frame-only rule.
    - Fallback mode: Polls {base_url}/shot.jpg over the SAME persistent HTTP Keep-Alive session with zero inter-frame sleep.
    - Fallback for local files / RTSP: cv2.VideoCapture with auto-looping.
    - Decoupled InferenceWorker: Runs YOLO inference (imgsz=320) asynchronously in the background.
    """
    def __init__(self, ip_url, fire_mod, base_mod, fire_conf, crowd_thresh, frame_skip):
        self.ip_url = ip_url
        self.fire_model = fire_mod
        self.base_model = base_mod
        self.fire_conf = fire_conf
        self.crowd_threshold = crowd_thresh
        self.frame_skip = frame_skip

        self.lock = threading.Lock()
        self.running = False
        self.capture_thread: threading.Thread | None = None
        self.session: requests.Session | None = None
        self.cap: cv2.VideoCapture | None = None

        self.latest_raw_frame: np.ndarray | None = None
        self.latest_frame: np.ndarray | None = None  # Backward-compatibility alias
        self.frame_seq: int = 0

        # Sub-worker for asynchronous YOLO inference
        self.inference_worker = InferenceWorker(
            fire_mod=fire_mod,
            base_mod=base_mod,
            fire_conf=fire_conf,
            crowd_thresh=crowd_thresh
        )

        # URL normalization for HTTP network cameras (e.g. IP Webcam or Cloudflare tunnel)
        self.is_http = str(self.ip_url).startswith("http://") or str(self.ip_url).startswith("https://")
        if self.is_http:
            clean_url = str(self.ip_url).rstrip("/")
            if clean_url.endswith("/video"):
                base = clean_url[:-6].rstrip("/")
            elif clean_url.endswith("/shot.jpg"):
                base = clean_url[:-9].rstrip("/")
            else:
                base = clean_url
            self.video_url = f"{base}/video"
            self.shot_url = f"{base}/shot.jpg"
        else:
            self.video_url = None
            self.shot_url = None

    @property
    def person_count(self) -> int:
        return self.inference_worker.person_count if self.inference_worker else 0

    @property
    def fire_spotted(self) -> bool:
        return self.inference_worker.fire_spotted if self.inference_worker else False

    @property
    def detections(self) -> list:
        return self.inference_worker.detections if self.inference_worker else []

    @property
    def inference_ms(self) -> float:
        return self.inference_worker.inference_ms if self.inference_worker else 0.0

    def get_raw_frame_and_seq(self) -> tuple[np.ndarray | None, int]:
        with self.lock:
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy(), self.frame_seq
            return None, -1

    def get_display_data(self) -> tuple[np.ndarray | None, list, int, bool, float]:
        with self.lock:
            raw = self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None

        if self.inference_worker:
            with self.inference_worker.lock:
                dets = list(self.inference_worker.detections)
                p_cnt = self.inference_worker.person_count
                fire_f = self.inference_worker.fire_spotted
                inf_ms = self.inference_worker.inference_ms
        else:
            dets, p_cnt, fire_f, inf_ms = [], 0, False, 0.0

        return raw, dets, p_cnt, fire_f, inf_ms

    def start(self):
        if self.running and self.capture_thread and self.capture_thread.is_alive():
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.inference_worker.start(self.get_raw_frame_and_seq)

    def stop(self):
        self.running = False
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.inference_worker is not None:
            self.inference_worker.stop()

        if self.capture_thread and self.capture_thread.is_alive() and self.capture_thread != threading.current_thread():
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None

    def _capture_loop(self):
        if self.is_http:
            self._capture_http_loop()
        else:
            self._capture_cv2_loop()

    def _capture_http_loop(self):
        target_url = self.shot_url
        if not target_url:
            return

        # Configure persistent requests.Session with connection pooling & Keep-Alive
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=1)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        headers = {
            "Connection": "keep-alive",
            "User-Agent": "HazardLens-NVR/2.0"
        }

        last_successful_fetch = time.time()
        while self.running:
            try:
                resp = self.session.get(
                    target_url,
                    timeout=(2.5, 2.5),
                    headers=headers
                )
                if resp.status_code == 200 and resp.content:
                    img_np = np.frombuffer(resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    if frame is not None and frame.size > 0:
                        with self.lock:
                            self.latest_raw_frame = frame
                            self.latest_frame = frame
                            self.frame_seq += 1
                        last_successful_fetch = time.time()
                else:
                    # Keep last valid frame on transient errors; only clear after >6s continuous failure
                    if time.time() - last_successful_fetch > 6.0:
                        with self.lock:
                            self.latest_raw_frame = None
                            self.latest_frame = None
            except Exception:
                # Keep last valid frame on transient network drops; only clear after >6s continuous failure
                if time.time() - last_successful_fetch > 6.0:
                    with self.lock:
                        self.latest_raw_frame = None
                        self.latest_frame = None

            time.sleep(0.03)

    def _capture_cv2_loop(self):
        try:
            self.cap = cv2.VideoCapture(self.ip_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.cap = None

        failed_reads = 0
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self.lock:
                        self.latest_raw_frame = frame
                        self.latest_frame = frame
                        self.frame_seq += 1
                    failed_reads = 0
                    time.sleep(0.03)  # ~30 FPS for local file/stream
                    continue

                # Auto-loop for local video files
                if isinstance(self.ip_url, str) and os.path.isfile(self.ip_url):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        with self.lock:
                            self.latest_raw_frame = frame
                            self.latest_frame = frame
                            self.frame_seq += 1
                        time.sleep(0.03)
                        continue

            failed_reads += 1
            time.sleep(0.08)
            if failed_reads >= 40:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                try:
                    self.cap = cv2.VideoCapture(self.ip_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                failed_reads = 0


# ---------------------------------------------------------
# --- FAST BGR/HUD OVERLAY & PRE-ENCODED JPEG RENDERING ---
# ---------------------------------------------------------
def draw_hud_and_detections(frame: np.ndarray, detections: list, inf_ms: float = 0.0) -> np.ndarray:
    """
    Renders cached YOLO bounding boxes and CCTV HUD overlay onto the raw camera frame.
    Runs on the display thread so video plays smoothly at high FPS even while YOLO runs asynchronously.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Bounding Boxes
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det.get("label", "")
        conf = det.get("conf", 0.0)
        det_type = det.get("type", "person")

        # Clamp box coordinates to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if det_type == "fire":
            box_color = (34, 34, 239)    # Bright red (BGR)
            badge_text = f"{label} {int(conf * 100)}%"
        else:
            box_color = (129, 185, 16)   # Emerald green (BGR)
            badge_text = f"PERSON {int(conf * 100)}%"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.40
        thickness = 1
        (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), box_color, -1)
        cv2.putText(annotated, badge_text, (x1 + 3, ty1 + th + 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 2. CCTV Timestamp Overlay (Top-Left)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cctv_text = f"REC · CAM 01 · {now_str}"
    cv2.putText(annotated, cctv_text, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(annotated, cctv_text, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (250, 250, 250), 1, cv2.LINE_AA)

    # 3. Latency / Telemetry Indicator (Top-Right)
    if inf_ms > 0:
        telemetry_text = f"YOLO {int(inf_ms)}ms"
        (tw2, _), _ = cv2.getTextSize(telemetry_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx = max(10, w - tw2 - 14)
        cv2.putText(annotated, telemetry_text, (tx, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, telemetry_text, (tx, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (16, 185, 129), 1, cv2.LINE_AA)

    return annotated

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Enterprise Control Console
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── STRONG TOP BRANDING ──
    armed_now = st.session_state.system_armed
    focal_color = "#10B981" if armed_now else "#71717A"

    st.markdown(f"""
    <div class="sb-brand">
      <div class="sb-badge">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="8.5" stroke="#FAFAFA" stroke-width="1.4"/>
          <circle cx="12" cy="12" r="5" stroke="#FAFAFA" stroke-width="1.2" stroke-opacity="0.6"/>
          <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22" stroke="#FAFAFA" stroke-width="1.4" stroke-linecap="round"/>
          <circle cx="12" cy="12" r="1.5" fill="{focal_color}"/>
        </svg>
      </div>
      <div class="sb-brand-text">
        <div class="sb-title-row">
          <span class="sb-title">
            <span class="word-hazard">Hazard</span><span class="word-lens">Lens</span>
          </span>
          <span class="sb-nvr-badge">NVR</span>
        </div>
        <div class="sb-subtitle">THREAT DETECTION CONSOLE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CAMERA STREAM INPUT (EMPTY DEFAULT, NOT PRE-FILLED) ──
    stream_url_input = st.text_input(
        "CAM 01 Stream URL",
        value="",
        placeholder="http://192.168.1.3:8080 or tunnel URL",
        key="cam_stream_url_input"
    )

    # ── SINGLE TOGGLE CONTROL BUTTON ──
    if not armed_now:
        if st.button("ARM SYSTEM", type="primary", width="stretch", key="btn_arm_system"):
            trimmed_input = stream_url_input.strip()
            target_url = None
            if trimmed_input:
                is_valid, validated_or_err = validate_stream_url(trimmed_input)
                if not is_valid:
                    st.error(f"Invalid URL: {validated_or_err}")
                else:
                    target_url = normalize_camera_url(validated_or_err)
                    save_camera_slot("CAM 01", target_url)
            else:
                # Fall back to saved valid URL in paired_cameras.json
                saved_url = resolve_paired_camera("CAM 01")
                if saved_url:
                    target_url = saved_url
                else:
                    st.error("Please enter a valid Camera Stream URL to connect.")

            if target_url:
                st.session_state.paired_camera_url = target_url
                st.session_state.system_armed = True

                old_worker = st.session_state.get('camera_worker')
                if old_worker is not None:
                    try:
                        old_worker.stop()
                    except Exception:
                        pass
                    st.session_state.camera_worker = None

                new_worker = CameraWorker(
                    ip_url=target_url,
                    fire_mod=fire_model,
                    base_mod=base_model,
                    fire_conf=FIRE_CONF_THRESHOLD,
                    crowd_thresh=CROWD_THRESHOLD,
                    frame_skip=FRAME_SKIP
                )
                st.session_state.camera_worker = new_worker
                new_worker.start()

                st.toast("System Armed · Monitoring Active")
                st.rerun()
    else:
        if st.button("DISARM SYSTEM", width="stretch", key="btn_disarm_system"):
            disarm_system()
            st.toast("System Disarmed · Standby Mode")
            st.rerun()

    # ── TELEMETRY & ESSENTIAL SETTINGS ──
    st.markdown(f"""
    <div class="sb-divider"></div>
    <div class="sb-section-title">Telemetry & Settings</div>
    <div class="sb-settings-grid">
      <div class="sb-setting-row">
        <span class="sb-setting-label">Crowd Limit</span>
        <span class="sb-setting-val">{CROWD_THRESHOLD} pers.</span>
      </div>
      <div class="sb-setting-row">
        <span class="sb-setting-label">Inference Model</span>
        <span class="sb-setting-val">Twin YOLOv8</span>
      </div>
      <div class="sb-setting-row">
        <span class="sb-setting-label">Voice Synthesis</span>
        <span class="sb-setting-val">Edge-TTS (HI-IN)</span>
      </div>
      <div class="sb-setting-row">
        <span class="sb-setting-label">Alert Cooldown</span>
        <span class="sb-setting-val">{EMAIL_COOLDOWN_SEC}s</span>
      </div>
      <div class="sb-setting-row">
        <span class="sb-setting-label">Silence Duration</span>
        <span class="sb-setting-val">{SILENCE_DURATION_SEC}s</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RESOLVE ACTIVE CAMERA FEED
# ─────────────────────────────────────────────────────────────────────────────
armed = st.session_state.system_armed
if armed:
    ip_url = st.session_state.get('paired_camera_url') or resolve_paired_camera("CAM 01") or ""
else:
    ip_url = ""

# ─────────────────────────────────────────────────────────────────────────────
# SLIM HEADER BAR
# ─────────────────────────────────────────────────────────────────────────────
worker_obj = st.session_state.get('camera_worker')
has_active_feed = (armed and worker_obj is not None and getattr(worker_obj, 'latest_frame', None) is not None)
active_cams = "1" if has_active_feed else "0"
status_pill_cls = "status-armed" if armed else "status-standby"
dot_cls = "dot-emerald" if armed else "dot-amber"
status_txt = "ARMED" if armed else "STANDBY"

st.markdown(f"""
<div class="hz-header-bar">
  <div class="hz-header-left">
    <span class="hz-title">Live Monitoring</span>
    <span class="hz-loc-tag">MIET Campus</span>
  </div>
  <div class="hz-header-right">
    <div class="pill">
      <span class="pill-label">Active</span>
      <span class="pill-val">{active_cams}/4</span>
    </div>
    <div class="pill">
      <span class="pill-label">Crowd Limit</span>
      <span class="pill-val">{CROWD_THRESHOLD}</span>
    </div>
    <div class="status-pill {status_pill_cls}">
      <span class="dot {dot_cls}"></span>
      <span>{status_txt}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP ALERT BANNER (FRAGMENT SCOPED — PERSISTENT ACROSS RERUNS)
# ─────────────────────────────────────────────────────────────────────────────
def render_alert_banner():
    """Renders the top alert banner persistently whenever an alert is active."""
    armed = st.session_state.get("system_armed", False)
    now = time.time()
    is_silenced = (now < st.session_state.get('silenced_until', 0.0))
    active_alert = st.session_state.get("active_alert")

    if not armed or is_silenced or not active_alert:
        return

    if active_alert == "FIRE":
        st.markdown("""
        <div class="a-banner a-fire">
          <div class="a-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6
                       .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0
                       c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>
            </svg>
          </div>
          <div class="a-content">
            <div class="a-kicker">Hazard Alert</div>
            <div class="a-ttl">Fire / Smoke Detected on CAM 01</div>
            <div class="a-msg">MIET College Main Gate · High-confidence AI detection · Automated dispatch active</div>
          </div>
        </div>""", unsafe_allow_html=True)
    elif active_alert == "CROWD":
        live_count = st.session_state.get("live_crowd_count", 0)
        st.markdown(f"""
        <div class="a-banner a-crwd">
          <div class="a-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
            </svg>
          </div>
          <div class="a-content">
            <div class="a-kicker">Occupancy Advisory</div>
            <div class="a-ttl">Crowd Limit Exceeded on CAM 01</div>
            <div class="a-msg">MIET College Main Gate · Live Count: {live_count} persons (Threshold: {CROWD_THRESHOLD})</div>
          </div>
        </div>""", unsafe_allow_html=True)

if hasattr(st, "fragment"):
    @st.fragment(run_every=0.5)
    def alert_banner_fragment():
        render_alert_banner()

    alert_banner_fragment()
else:
    render_alert_banner()

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA GRID (2x2 ENTERPRISE NVR LAYOUT)
# ─────────────────────────────────────────────────────────────────────────────
live_badge    = '<span class="badge-status badge-live"><span class="dot dot-emerald"></span>LIVE</span>'
warn_badge    = '<span class="badge-status badge-warn"><span class="dot dot-amber"></span>CONNECTING</span>'
offline_badge = '<span class="badge-status badge-offline"><span class="dot dot-muted"></span>OFFLINE</span>'

def render_cam_01_feed(stream_url):
    """Dynamic feed, card top bar, contextual silence, and hazard detection for CAM 01."""
    if st.session_state.pop("silence_toast_pending", False):
        st.toast(f"CAM 01 alerts silenced for {SILENCE_DURATION_SEC}s.")

    armed = st.session_state.get("system_armed", False)
    now = time.time()
    is_silenced = (now < st.session_state.get('silenced_until', 0.0))

    # When disarmed, ensure worker stopped and render clean standby state
    if not armed:
        if st.session_state.get('camera_worker') is not None:
            try:
                st.session_state.camera_worker.stop()
            except Exception:
                pass
            st.session_state.camera_worker = None

        st.session_state.active_alert = None
        st.session_state.hazard_clear_ts = None
        st.session_state.active_audio_b64 = None
        st.session_state.audio_expire_ts = 0.0
        play_persistent_audio(None, 0)

        st.markdown(f"""
        <div class="cam-card-hdr">
          <div class="cam-card-meta">
            <span class="cam-id">CAM 01</span>
            <span class="cam-name">MIET College Main Gate</span>
          </div>
          <div class="cam-card-right">
            <span class="cam-crowd-count">— / {CROWD_THRESHOLD}</span>
            {offline_badge}
          </div>
        </div>""", unsafe_allow_html=True)
        st.image(create_placeholder("CAM 01", "STANDBY"), width="stretch")
        return

    # If stream_url is not configured yet
    if not stream_url:
        st.markdown(f"""
        <div class="cam-card-hdr">
          <div class="cam-card-meta">
            <span class="cam-id">CAM 01</span>
            <span class="cam-name">MIET College Main Gate</span>
          </div>
          <div class="cam-card-right">
            <span class="cam-crowd-count">— / {CROWD_THRESHOLD}</span>
            {offline_badge}
          </div>
        </div>""", unsafe_allow_html=True)
        st.image(create_placeholder("CAM 01", "NO SIGNAL"), width="stretch")
        return

    # When armed, ensure background worker is running for stream_url
    worker = st.session_state.get('camera_worker')
    if worker is None or not worker.running or worker.ip_url != stream_url:
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        worker = CameraWorker(
            ip_url=stream_url,
            fire_mod=fire_model,
            base_mod=base_model,
            fire_conf=FIRE_CONF_THRESHOLD,
            crowd_thresh=CROWD_THRESHOLD,
            frame_skip=FRAME_SKIP
        )
        st.session_state.camera_worker = worker
        worker.start()

    # Read latest raw frame and async YOLO detection state from worker
    raw_frame, detections, person_count, fire_spotted, inf_ms = worker.get_display_data()

    st.session_state.live_crowd_count = person_count

    has_feed = (raw_frame is not None)
    badge1 = live_badge if has_feed else warn_badge
    crowd_str = f"{person_count} / {CROWD_THRESHOLD}"

    st.markdown(f"""
    <div class="cam-card-hdr">
      <div class="cam-card-meta">
        <span class="cam-id">CAM 01</span>
        <span class="cam-name">MIET College Main Gate</span>
      </div>
      <div class="cam-card-right">
        <span class="cam-crowd-count">{crowd_str}</span>
        {badge1}
      </div>
    </div>""", unsafe_allow_html=True)

    if raw_frame is not None:
        annotated = draw_hud_and_detections(raw_frame, detections, inf_ms)
        # Fast JPEG bytes encoding for zero Streamlit conversion overhead
        ret, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret and jpeg_buf is not None and len(jpeg_buf) > 0:
            st.image(jpeg_buf.tobytes(), width="stretch")
        else:
            st.image(annotated, channels="BGR", width="stretch")
    else:
        st.image(create_placeholder("CAM 01", "CONNECTING"), width="stretch")
        if str(stream_url).startswith("http"):
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:11px;color:var(--amber);text-align:center;padding:6px 0;">'
                f'Awaiting feed ({stream_url}) &middot; Ensure camera server is running</div>',
                unsafe_allow_html=True
            )

    # ── HAZARD DETECTION ENGINE ("Latest Hazard Wins" + Silence Cooldown + Debounced Clearing) ──
    if is_silenced:
        st.session_state.active_alert = None
        st.session_state.active_audio_b64 = None
        st.session_state.audio_expire_ts = 0.0
        play_persistent_audio(None, 0)
        # Debounced Auto-Release
        if not fire_spotted and person_count <= CROWD_THRESHOLD:
            if st.session_state.hazard_clear_ts is None:
                st.session_state.hazard_clear_ts = now
            elif now - st.session_state.hazard_clear_ts >= ALERT_CLEAR_GRACE_SEC:
                st.session_state.silenced_until  = 0.0
                st.session_state.hazard_clear_ts = None
        else:
            st.session_state.hazard_clear_ts = None
    else:
        if fire_spotted:
            st.session_state.hazard_clear_ts = None
            prev_alert = st.session_state.get("active_alert")
            if prev_alert != "FIRE":
                last_audio = st.session_state.last_audio_ts.get("FIRE", 0.0)
                if now - last_audio > AUDIO_COOLDOWN_SEC:
                    trigger_hindi_alert(FIRE_TTS_TEXT)
                    st.session_state.last_audio_ts["FIRE"] = now
                email_alerter.send_alert("FIRE")
            st.session_state.active_alert = "FIRE"

        elif person_count > CROWD_THRESHOLD:
            st.session_state.hazard_clear_ts = None
            prev_alert = st.session_state.get("active_alert")
            if prev_alert != "CROWD":
                last_audio = st.session_state.last_audio_ts.get("CROWD", 0.0)
                if now - last_audio > AUDIO_COOLDOWN_SEC:
                    trigger_hindi_alert(CROWD_TTS_TEXT)
                    st.session_state.last_audio_ts["CROWD"] = now
                email_alerter.send_alert("CROWD", person_count)
            st.session_state.active_alert = "CROWD"

        else:
            # Reset alert display if hazards have cleared (debounced with grace period to prevent flickering)
            if st.session_state.get("active_alert") is not None:
                if st.session_state.hazard_clear_ts is None:
                    st.session_state.hazard_clear_ts = now
                elif now - st.session_state.hazard_clear_ts >= ALERT_CLEAR_GRACE_SEC:
                    st.session_state.active_alert = None
                    st.session_state.hazard_clear_ts = None

    # Persistent Web Audio Playback (Loops infinitely until Silence/Disarm)
    active_b64 = st.session_state.get('active_audio_b64')
    active_id = st.session_state.get('active_audio_id', 0)

    if armed and not is_silenced and active_b64:
        play_persistent_audio(active_b64, active_id)
    else:
        if not armed or is_silenced:
            st.session_state.active_audio_b64 = None
            play_persistent_audio(None, 0)

    # Contextual Silence Button: Only rendered when an alert is actively ringing on CAM 01
    if armed and not is_silenced and st.session_state.get("active_alert") is not None:
        st.button("Silence Alert", key="silence_1", on_click=silence_alerts, width="stretch")

col1, col2 = st.columns(2)

with col1:
    if hasattr(st, "fragment"):
        @st.fragment(run_every=0.15)
        def cam_01_live_fragment(stream_url):
            render_cam_01_feed(stream_url)

        cam_01_live_fragment(ip_url)
    else:
        render_cam_01_feed(ip_url)
        if st.session_state.system_armed:
            time.sleep(0.15)
            st.rerun()

with col2:
    st.markdown(f"""
    <div class="cam-card-hdr">
      <div class="cam-card-meta">
        <span class="cam-id">CAM 02</span>
        <span class="cam-name">Railway Station Entrance</span>
      </div>
      <div class="cam-card-right">
        <span class="cam-crowd-count">— / {CROWD_THRESHOLD}</span>
        {offline_badge}
      </div>
    </div>""", unsafe_allow_html=True)
    st.image(create_placeholder("CAM 02", "STANDBY"), width="stretch")

col3, col4 = st.columns(2)

with col3:
    st.markdown(f"""
    <div class="cam-card-hdr">
      <div class="cam-card-meta">
        <span class="cam-id">CAM 03</span>
        <span class="cam-name">Metro Station Entrance</span>
      </div>
      <div class="cam-card-right">
        <span class="cam-crowd-count">— / {CROWD_THRESHOLD}</span>
        {offline_badge}
      </div>
    </div>""", unsafe_allow_html=True)
    st.image(create_placeholder("CAM 03", "STANDBY"), width="stretch")

with col4:
    st.markdown(f"""
    <div class="cam-card-hdr">
      <div class="cam-card-meta">
        <span class="cam-id">CAM 04</span>
        <span class="cam-name">Server Room Backend</span>
      </div>
      <div class="cam-card-right">
        <span class="cam-crowd-count">— / {CROWD_THRESHOLD}</span>
        {offline_badge}
      </div>
    </div>""", unsafe_allow_html=True)
    st.image(create_placeholder("CAM 04", "STANDBY"), width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER STATUS BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
  <span>HAZARDLENS · MIET CAMPUS</span>
  <span>YOLOV8 · EDGE-TTS · RESEND</span>
  <span>ALL TIMES IST (UTC+05:30)</span>
  <span id="ftr-dt">—</span>
</div>
<script>
(function(){
  var el = document.getElementById('ftr-dt');
  if(el) el.textContent = new Date().toLocaleDateString('en-IN',
    {weekday:'short', year:'numeric', month:'short', day:'numeric'});
})();
</script>
""", unsafe_allow_html=True)
