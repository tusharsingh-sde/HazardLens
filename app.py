import streamlit as st
import streamlit.components.v1 as components
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import edge_tts
import asyncio
import os
import hashlib
import time
import threading
import urllib.request
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
    page_title="HazardLens | Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg:    #0B0F14;
  --s1:    #111820;
  --s2:    #17212C;
  --line:  #1F2B38;
  --line2: #2D3C4D;
  --text:  #E6EDF3;
  --text2: #A3B3C2;
  --mute:  #7B8DA0;
  --acc:   #3CC8E8;
  --crit:  #FF5C4D;
  --warn:  #F2B33D;
  --ok:    #3DD68C;
  --mono:  'IBM Plex Mono', ui-monospace, monospace;
  --sans:  'IBM Plex Sans', system-ui, sans-serif;
  --cond:  'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif;
}

/* ── BASE ── */
.stApp, .main { background: var(--bg) !important; color: var(--text) !important; }
#MainMenu, footer { visibility: hidden !important; }
*, *::before, *::after { box-sizing: border-box; }
div.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: var(--s1) !important;
  border-right: 1px solid var(--line) !important;
}

/* ── HEADER ── */
.hz-hdr {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px 28px;
  padding: 10px 24px;
  background: var(--s1);
  border: 1px solid var(--line);
  border-radius: 4px;
  margin-bottom: 12px;
}
.hz-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.hz-logo { flex: 0 0 auto; }
.hz-name {
  font: 600 20px/1 var(--cond);
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--text);
}
.hz-name em { color: var(--acc); font-style: normal; }
.hz-sub {
  font: 500 11px/1.2 var(--mono);
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--mute);
  margin-top: 3px;
}
.hz-right { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-left: auto; }

.stat-pill {
  background: var(--s2);
  border: 1px solid var(--line2);
  border-radius: 3px;
  padding: 6px 14px;
  text-align: center;
  min-width: 72px;
}
.stat-v { display: block; font: 500 20px/1 var(--mono); color: var(--acc); }
.stat-l { font: 500 11px/1.2 var(--mono); color: var(--mute); letter-spacing: .1em; text-transform: uppercase; }

.arm-on {
  display: inline-flex; align-items: center; gap: 8px;
  height: 44px; padding: 0 18px;
  border: 1px solid var(--ok); border-radius: 3px;
  background: rgba(61,214,140,.08);
  font: 600 13px var(--sans); color: var(--ok);
  letter-spacing: .02em;
}
.arm-off {
  display: inline-flex; align-items: center; gap: 8px;
  height: 44px; padding: 0 18px;
  border: 1px solid var(--line2); border-radius: 3px;
  background: var(--s2);
  font: 600 13px var(--sans); color: var(--mute);
  letter-spacing: .02em;
}

/* PULSING DOTS */
@keyframes hzPulse { 0%,100%{opacity:1} 50%{opacity:.25} }
@keyframes hzPulseFast { 0%,100%{opacity:1} 50%{opacity:.1} }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-g { background: var(--ok);   animation: hzPulse 1.4s ease-in-out infinite; }
.dot-a { background: var(--warn); animation: hzPulse 1.4s ease-in-out infinite; }
.dot-r { background: var(--crit); animation: hzPulseFast .7s ease-in-out infinite; }

/* ── ALERT BANNERS ── */
@keyframes si { from{transform:translateY(-6px);opacity:0;} to{transform:translateY(0);opacity:1;} }

.a-fire {
  display: flex; flex-wrap: wrap; align-items: center;
  gap: 16px 28px; padding: 16px 24px; margin: 6px 0;
  background: rgba(255,92,77,.08);
  border: 1px solid var(--crit);
  border-radius: 4px;
  animation: si .25s ease;
}
.a-crwd {
  display: flex; flex-wrap: wrap; align-items: center;
  gap: 16px 28px; padding: 16px 24px; margin: 6px 0;
  background: rgba(242,179,61,.08);
  border: 1px solid var(--warn);
  border-radius: 4px;
  animation: si .25s ease;
}
.a-ico-wrap {
  flex: 0 0 auto; width: 48px; height: 48px;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid currentColor; border-radius: 2px;
}
.a-fire .a-ico-wrap { color: var(--crit); }
.a-crwd .a-ico-wrap { color: var(--warn); }
.a-kicker {
  font: 500 11px/1.2 var(--mono);
  letter-spacing: .12em; text-transform: uppercase;
}
.a-fire .a-kicker { color: var(--crit); }
.a-crwd .a-kicker { color: var(--warn); }
.a-ttl {
  margin-top: 4px;
  font: 600 20px/1.15 var(--cond);
  letter-spacing: .01em;
  color: var(--text);
}
.a-msg { margin-top: 4px; font-size: 13px; color: var(--text2); }

/* ── SECTION LABEL ── */
.sec-lbl {
  font: 500 11px/1.2 var(--mono);
  letter-spacing: .1em; text-transform: uppercase;
  color: var(--mute);
  margin: 14px 0 8px;
  display: flex; align-items: center; gap: 10px;
}
.sec-lbl::after { content:''; flex:1; height:1px; background: var(--line); }

/* ── CAMERA CARD HEADER ── */
.cam-hdr {
  background: var(--s2);
  border: 1px solid var(--line); border-bottom: none;
  border-radius: 4px 4px 0 0;
  padding: 10px 16px;
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
}
.cam-hdr.live { border-color: var(--acc); }
.cam-nm  { font: 500 12px/1.2 var(--mono); letter-spacing: .1em; color: var(--acc); }
.cam-loc { margin-top: 2px; font: 500 13px/1.2 var(--sans); color: var(--text); }

.badge-live  {
  display: inline-flex; align-items: center; gap: 6px;
  height: 26px; padding: 0 10px;
  border: 1px solid var(--crit); border-radius: 3px;
  font: 500 11px/1 var(--mono); color: var(--crit); letter-spacing: .1em;
}
.badge-stdby {
  display: inline-flex; align-items: center; gap: 6px;
  height: 26px; padding: 0 10px;
  border: 1px solid var(--line2); border-radius: 3px;
  font: 500 11px/1 var(--mono); color: var(--mute); letter-spacing: .08em;
}

/* ── METRIC STRIP ── */
.mstrip {
  background: var(--s2); border: 1px solid var(--line); border-top: none; border-bottom: none;
  padding: 8px 16px; display: flex; align-items: center; justify-content: space-between;
}
.mstrip .ml { font: 500 11px/1.2 var(--mono); letter-spacing: .1em; text-transform: uppercase; color: var(--mute); }
.mstrip .mv { font: 500 20px/1 var(--mono); color: var(--acc); }
.mstrip .mv.off { color: var(--mute); font-size: 15px; }

/* ── IMAGE ── */
div[data-testid="stImage"] { margin: 0 !important; padding: 0 !important; position: relative; }
div[data-testid="stImage"] img {
  display: block; width: 100%;
  border: 1px solid var(--line); border-top: none; border-radius: 0;
}
div[data-testid="stImage"]::after {
  content: '';
  position: absolute; inset: 0;
  background: repeating-linear-gradient(0deg, rgba(255,255,255,.018) 0 1px, transparent 1px 3px);
  pointer-events: none; z-index: 5;
}

/* ── BUTTONS ── */
div[data-testid="stButton"] > button {
  font-family: var(--sans) !important;
  font-size: 13px !important; font-weight: 600 !important;
  letter-spacing: .02em !important;
  border-radius: 0 0 4px 4px !important;
  border: 1px solid var(--line) !important; border-top: none !important;
  width: 100% !important;
  background: var(--s2) !important; color: var(--text2) !important;
  padding: 10px 12px !important; transition: all .15s ease !important;
}
div[data-testid="stButton"] > button:hover {
  background: var(--s1) !important; color: var(--acc) !important;
  border-color: var(--acc) !important;
}

/* ── TEXT INPUT ── */
div[data-testid="stTextInput"] input {
  background: var(--s2) !important; border: 1px solid var(--line) !important;
  color: var(--acc) !important; font: 400 13px var(--mono) !important;
  letter-spacing: .04em !important; border-radius: 3px !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--acc) !important; box-shadow: 0 0 0 1px rgba(60,200,232,.25) !important;
}
div[data-testid="stTextInput"] label {
  font: 500 11px var(--mono) !important;
  color: var(--mute) !important; letter-spacing: .1em !important; text-transform: uppercase !important;
}

/* ── SIDEBAR COMPONENTS ── */
.sb-title {
  font: 600 15px/1.2 var(--cond); color: var(--acc); letter-spacing: .06em;
  text-transform: uppercase; padding-bottom: 12px; margin-bottom: 12px;
  border-bottom: 1px solid var(--line);
}
.sb-lbl { font: 500 11px/1.2 var(--mono); color: var(--mute); letter-spacing: .1em; text-transform: uppercase; padding: 12px 0 7px; }
.sb-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 12px; margin: 3px 0;
  background: var(--s2); border: 1px solid var(--line); border-radius: 3px;
}
.sb-rn    { font: 500 13px var(--sans); color: var(--text); }
.sb-rloc  { font: 400 12px var(--mono); color: var(--mute); letter-spacing: .04em; margin-top: 2px; }
.sb-rs    { font: 500 11px var(--mono); letter-spacing: .08em; }
.sysinfo  {
  font: 400 12px/2 var(--mono); color: var(--mute);
  padding: 10px 14px; background: var(--s2); border: 1px solid var(--line); border-radius: 3px;
}

/* ── FOOTER STATUS BAR ── */
.ftr {
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 24px;
  padding: 12px 24px; margin-top: 16px;
  border-top: 1px solid var(--line); border: 1px solid var(--line);
  background: var(--s1); border-radius: 4px;
  font: 400 11px/1.4 var(--mono); letter-spacing: .06em; color: var(--mute);
}
.ftr .online { color: var(--ok); }

/* ── TOAST ── */
div[data-testid="stToast"] {
  background: rgba(61,214,140,.08) !important; border: 1px solid var(--ok) !important;
  color: var(--ok) !important; font-family: var(--sans) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--line2); border-radius: 2px; }

/* Sidebar buttons — no bottom-only radius */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
  border-radius: 3px !important;
  border: 1px solid var(--line) !important;
  border-top: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
  border-color: var(--acc) !important;
}

/* Keep background alert audio player hidden so it does not distort camera grid layout */
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
        st.error(f"🚨 CRITICAL SECURITY ALERT: Model integrity verification failed: {val}")
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
    """Tactical-style placeholder with reticle, grid, and corner brackets."""
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8) + 11

    # Grid lines
    for x in range(0, w, 64):
        cv2.line(img, (x, 0), (x, h), (20, 33, 44), 1)
    for y in range(0, h, 64):
        cv2.line(img, (0, y), (w, y), (20, 33, 44), 1)

    # Reticle
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 50, cy), (cx - 14, cy), (31, 56, 72), 1)
    cv2.line(img, (cx + 14, cy), (cx + 50, cy), (31, 56, 72), 1)
    cv2.line(img, (cx, cy - 40), (cx, cy - 14), (31, 56, 72), 1)
    cv2.line(img, (cx, cy + 14), (cx, cy + 40), (31, 56, 72), 1)
    cv2.circle(img, (cx, cy), 20, (31, 56, 72), 1)
    cv2.circle(img, (cx, cy), 55, (22, 40, 55), 1)
    cv2.circle(img, (cx, cy), 3,  (35, 60, 80), -1)

    # Corner brackets
    bs = 20
    corners = [(40, 30, 1, 1), (w - 40, 30, -1, 1),
               (40, h - 30, 1, -1), (w - 40, h - 30, -1, -1)]
    for bx, by, dx, dy in corners:
        cv2.line(img, (bx, by), (bx + bs * dx, by), (60, 200, 232), 1)
        cv2.line(img, (bx, by), (bx, by + bs * dy), (60, 200, 232), 1)

    lw = cv2.getTextSize(cam_name, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
    cv2.putText(img, cam_name, (cx - lw // 2, cy - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 200, 232), 1)
    sw = cv2.getTextSize(f"[ {status} ]", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    cv2.putText(img, f"[ {status} ]", (cx - sw // 2, cy + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (45, 70, 90), 1)

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
    import streamlit.deprecation_util as _deprec_util
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

def play_persistent_audio(b64_audio: str = None, audio_id: int = 0, force: bool = False):
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
# --- THREADED CAMERA WORKER (NON-BLOCKING INFERENCE) ---
# ---------------------------------------------------------
class CameraWorker:
    """Background worker for non-blocking video capture and YOLO inference."""
    def __init__(self, ip_url, fire_mod, base_mod, fire_conf, crowd_thresh, frame_skip):
        self.ip_url = ip_url
        self.fire_model = fire_mod
        self.base_model = base_mod
        self.fire_conf = fire_conf
        self.crowd_threshold = crowd_thresh
        self.frame_skip = frame_skip

        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cap = None

        self.latest_frame = None
        self.person_count = 0
        self.fire_spotted = False

        # Fast HTTP snapshot grabber for IP Webcam / network cameras
        self.is_http = str(self.ip_url).startswith("http://") or str(self.ip_url).startswith("https://")
        if self.is_http:
            clean_url = str(self.ip_url).rstrip("/")
            if clean_url.endswith("/video"):
                clean_base = clean_url[:-6].rstrip("/")
                self.shot_url = f"{clean_base}/shot.jpg"
            elif clean_url.endswith("/shot.jpg"):
                self.shot_url = clean_url
            else:
                self.shot_url = f"{clean_url}/shot.jpg"
        else:
            self.shot_url = None

    def start(self):
        if self.running and self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.thread and self.thread.is_alive() and self.thread != threading.current_thread():
            self.thread.join(timeout=1.0)
        self.thread = None

    def _read_frame(self):
        """
        Grab a fresh frame.
        For HTTP streams: Fast /shot.jpg HTTP fetcher bypasses OpenCV MSMF/MJPEG bugs with zero lag.
        For local video files / RTSP: Uses cv2.VideoCapture with auto-looping.
        """
        # Primary for HTTP network cameras: /shot.jpg HTTP snapshot
        if self.is_http and self.shot_url:
            try:
                req = urllib.request.Request(self.shot_url, headers={"User-Agent": "HazardLens-FrameGrabber/1.0"})
                with urllib.request.urlopen(req, timeout=1.2) as resp:
                    if resp.status == 200:
                        img_bytes = resp.read()
                        if img_bytes:
                            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                            if frame is not None and frame.size > 0:
                                return True, frame
            except Exception:
                pass

        # Fallback to cv2.VideoCapture (for local MP4, RTSP, or fallback)
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame
            # Infinite loop for local demo video files
            if isinstance(self.ip_url, str) and os.path.isfile(self.ip_url):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame

        return False, None

    def _worker_loop(self):
        # Only initialize cv2.VideoCapture for local files or RTSP streams!
        # For HTTP network cameras (e.g. IP Webcam), pure HTTP frame grabber is used
        # to avoid OpenCV FFMPEG TCP connection errors (-138) and thread blocking.
        if not self.is_http:
            try:
                self.cap = cv2.VideoCapture(self.ip_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                self.cap = None

        count = 0
        failed_reads = 0
        while self.running:
            ret, frame = self._read_frame()
            if not ret or frame is None:
                failed_reads += 1
                time.sleep(0.08)
                if failed_reads >= 50 and not self.is_http:  # Re-attempt VideoCapture initialization after dropped file stream
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
                continue

            failed_reads = 0
            count += 1
            if count % self.frame_skip == 0:
                try:
                    with model_inference_lock:
                        # Engine 1: Custom Fire / Smoke model (imgsz=416, verbose=False for fast edge inference)
                        fire_results = self.fire_model(frame, conf=self.fire_conf, imgsz=416, verbose=False)
                        annotated_frame = fire_results[0].plot()
                        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                        # CCTV Timestamp Overlay
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cctv_text = f"REC · CAM 01 · {now_str}"
                        cv2.putText(annotated_rgb, cctv_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(annotated_rgb, cctv_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 200, 232), 2, cv2.LINE_AA)

                        # Engine 2: Base model for crowd counting (imgsz=416, verbose=False)
                        base_results = self.base_model(frame, imgsz=416, verbose=False)
                        p_count = sum(
                            1 for box in base_results[0].boxes
                            if self.base_model.names[int(box.cls[0])] == 'person'
                        )

                        fire_found = any(
                            self.fire_model.names[int(b.cls[0])].lower() in ('fire', 'smoke', '火', '烟')
                            for b in fire_results[0].boxes
                        )

                    with self.lock:
                        self.latest_frame = annotated_rgb
                        self.person_count = p_count
                        self.fire_spotted = fire_found
                except Exception as e:
                    print(f"Inference error in worker: {e}")
            else:
                time.sleep(0.005)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Control Panel
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">⬡ Control Panel</div>', unsafe_allow_html=True)

    # ── TELEMETRY CHANNEL & HARDWARE REGISTRY ──
    st.markdown('<div class="sb-lbl">▸ Telemetry Channel</div>', unsafe_allow_html=True)
    feed_mode = st.radio(
        "Telemetry Channel",
        ["📡 Paired CCTV Network (Auto-Sync)", "🎬 Tactical Demo Simulation"],
        index=0,
        label_visibility="collapsed"
    )

    armed_now = st.session_state.system_armed
    if feed_mode == "🎬 Tactical Demo Simulation":
        ip_url = "assets/demo_feed.mp4"
        st.markdown(
            '<div style="font-family:var(--mono);font-size:11px;color:var(--acc);'
            'padding:6px 10px;background:var(--s2);border:1px solid var(--line2);border-radius:3px;margin-bottom:8px;">'
            '🎬 TACTICAL DEMO ACTIVE (Simulation)</div>',
            unsafe_allow_html=True
        )
    else:
        if armed_now:
            # 1-Click Auto-IP Healing via Dynamic Registry
            resolved_url = resolve_paired_camera("CAM 01")
            if resolved_url:
                st.session_state.paired_camera_url = resolved_url
                ip_url = resolved_url
                st.markdown(
                    '<div style="font-family:var(--mono);font-size:11px;color:var(--ok);'
                    'padding:6px 10px;background:rgba(61,214,140,0.08);border:1px solid var(--ok);border-radius:3px;margin-bottom:8px;">'
                    '🟢 CAM 01 ONLINE (Auto-Synced)</div>',
                    unsafe_allow_html=True
                )
            else:
                ip_url = "assets/demo_feed.mp4"
                st.markdown(
                    '<div style="font-family:var(--mono);font-size:11px;color:var(--warn);'
                    'padding:6px 10px;background:rgba(242,179,61,0.08);border:1px solid var(--warn);border-radius:3px;margin-bottom:8px;">'
                    '🟡 CAM 01 AUTO-HEAL: SCANNING (Demo Mode)</div>',
                    unsafe_allow_html=True
                )
        else:
            ip_url = "assets/demo_feed.mp4"
            cam_info = get_paired_camera("CAM 01")
            last_ip = cam_info.get("last_ip_port") if (cam_info and cam_info.get("last_ip_port")) else "Unpaired"
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:11px;color:var(--mute);'
                f'padding:6px 10px;background:var(--s2);border:1px solid var(--line2);border-radius:3px;margin-bottom:8px;">'
                f'⚪ CAM 01 STANDBY ({last_ip})</div>',
                unsafe_allow_html=True
            )

    # ── 2-STEP CAMERA CONNECTION FLOW (CAM 01) ──
    st.markdown('<div class="sb-lbl">▸ CAM 01 Stream Connection</div>', unsafe_allow_html=True)
    cam_info = get_paired_camera("CAM 01")
    current_saved_url = ""
    if cam_info:
        current_saved_url = cam_info.get("stream_url") or cam_info.get("last_ip_port") or ""
        if current_saved_url and not current_saved_url.startswith(("http://", "https://")):
            current_saved_url = f"http://{current_saved_url}"

    stream_url_input = st.text_input(
        "Camera Stream URL (Local IP or Cloudflare Tunnel)",
        value=current_saved_url,
        placeholder="http://192.168.1.3:8080 or https://xxx.trycloudflare.com",
        help="Paste local IP Webcam URL or Cloudflare Tunnel endpoint",
        key="cam_stream_url_input"
    )

    btn_connect_arm = st.button("🔴 Connect & ARM", width="stretch", key="btn_connect_arm")

    if btn_connect_arm:
        trimmed_input = stream_url_input.strip()
        if not trimmed_input:
            st.error("Please enter a valid Camera Stream URL.")
        else:
            is_valid, validated_or_err = validate_stream_url(trimmed_input)
            if not is_valid:
                st.error(f"Invalid URL: {validated_or_err}")
            else:
                normalized_url = normalize_camera_url(validated_or_err)
                save_camera_slot("CAM 01", normalized_url)
                st.session_state.paired_camera_url = normalized_url
                st.session_state.system_armed = True

                # Immediately start CameraWorker with the new normalized URL
                old_worker = st.session_state.get('camera_worker')
                if old_worker is not None:
                    try:
                        old_worker.stop()
                    except Exception:
                        pass
                    st.session_state.camera_worker = None

                new_worker = CameraWorker(
                    ip_url=normalized_url,
                    fire_mod=fire_model,
                    base_mod=base_model,
                    fire_conf=FIRE_CONF_THRESHOLD,
                    crowd_thresh=CROWD_THRESHOLD,
                    frame_skip=FRAME_SKIP
                )
                st.session_state.camera_worker = new_worker
                new_worker.start()

                st.toast("✅ CAM 01 Connected & System Armed!")
                st.rerun()

    # ── SYSTEM CONTROL (1-CLICK TOGGLE) ──
    st.markdown('<div class="sb-lbl">▸ System Control</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.button("🔴 ARM",    on_click=arm_system,    width="stretch")
    with cb:
        st.button("⬛ DISARM", on_click=disarm_system, width="stretch")

    st.markdown('<div class="sb-lbl">▸ Camera Status</div>', unsafe_allow_html=True)
    cam_registry = [
        ("CAM 01", "MIET COLLEGE MAIN GATE",   True),
        ("CAM 02", "RAILWAY STATION ENTRANCE",  False),
        ("CAM 03", "METRO STATION ENTRANCE",    False),
        ("CAM 04", "SERVER ROOM BACKEND",       False),
    ]
    for cn, loc, active in cam_registry:
        is_live  = active and armed_now
        s_text   = "LIVE"  if is_live else "STDBY"
        s_color  = "var(--ok)" if is_live else "var(--mute)"
        dot_cls  = "dot dot-r" if is_live else "dot dot-a"
        st.markdown(f"""
        <div class="sb-row">
          <div>
            <div class="sb-rn">{cn}</div>
            <div class="sb-rloc">{loc}</div>
          </div>
          <span class="sb-rs" style="color:{s_color};">
            <span class="{dot_cls}" style="width:7px;height:7px;margin-right:4px;"></span>{s_text}
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-lbl">▸ System Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sysinfo">
    LOCATION &nbsp;&nbsp; : MIET CAMPUS<br>
    PAIRING &nbsp;&nbsp;&nbsp; : AUTO (CRYPTOGRAPHIC)<br>
    AUTH &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : MUTUAL BASIC-AUTH<br>
    INTEGRITY &nbsp; : SHA-256 VERIFIED<br>
    AI ENGINES : TWIN (FIRE + CROWD)<br>
    TTS ENGINE : EDGE-TTS (HINDI NEURAL)<br>
    LANGUAGE &nbsp;&nbsp; : HI-IN<br>
    FRAME SKIP : {FRAME_SKIP} FRAMES<br>
    CROWD LIM &nbsp;: {CROWD_THRESHOLD} PERSONS<br>
    CONF THRESH: {FIRE_CONF_THRESHOLD}<br>
    COOLDOWN &nbsp;&nbsp;: {EMAIL_COOLDOWN_SEC}s<br>
    SILENCE &nbsp;&nbsp;&nbsp;: {SILENCE_DURATION_SEC}s
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MASTER HEADER
# ─────────────────────────────────────────────────────────────────────────────
armed = st.session_state.system_armed
arm_cls = "arm-on"  if armed else "arm-off"
dot_cls = "dot dot-g" if armed else "dot dot-a"
arm_txt = "ARMED"   if armed else "STANDBY"

st.markdown(f"""
<div class="hz-hdr">
  <div class="hz-brand">
    <svg class="hz-logo" width="28" height="28" viewBox="0 0 24 24" fill="none"
         stroke="#3CC8E8" stroke-width="1.6" stroke-linejoin="round" aria-hidden="true">
      <path d="M12 2.5 20.5 7.25v9.5L12 21.5l-8.5-4.75v-9.5z"/>
      <circle cx="12" cy="12" r="3.2"/>
    </svg>
    <div>
      <div class="hz-name">Hazard<em>Lens</em></div>
      <div class="hz-sub">AI-Powered Smart City Surveillance Command · MIET Campus</div>
    </div>
  </div>
  <div class="hz-right">
    <div class="stat-pill">
      <span class="stat-v">04</span>
      <span class="stat-l">Cameras</span>
    </div>
    <div class="stat-pill">
      <span class="stat-v">{CROWD_THRESHOLD}</span>
      <span class="stat-l">Crowd Lim</span>
    </div>
    <div class="stat-pill">
      <span class="stat-v" style="font-size:13px;color:var(--ok);">AUTO</span>
      <span class="stat-l">Pairing</span>
    </div>
    <div class="stat-pill">
      <span class="stat-v" style="font-size:13px;color:var(--ok);">OP</span>
      <span class="stat-l">Access</span>
    </div>
    <div class="{arm_cls}">
      <span class="{dot_cls}"></span>{arm_txt}
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
        <div class="a-fire">
          <div class="a-ico-wrap">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.8"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6
                       .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0
                       c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>
            </svg>
          </div>
          <div>
            <div class="a-kicker">FIRE · SMOKE HAZARD</div>
            <div class="a-ttl">Fire / Smoke Detected — CAM 01</div>
            <div class="a-msg">MIET Main Gate · High-confidence AI detection · Fire Safety Team Dispatched</div>
          </div>
        </div>""", unsafe_allow_html=True)
    elif active_alert == "CROWD":
        live_count = st.session_state.get("live_crowd_count", 0)
        st.markdown(f"""
        <div class="a-crwd">
          <div class="a-ico-wrap">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.8"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
            </svg>
          </div>
          <div>
            <div class="a-kicker">OVERCROWDING ALERT</div>
            <div class="a-ttl">Crowd Limit Exceeded — CAM 01</div>
            <div class="a-msg">MIET Main Gate · Live Count: {live_count} persons · Security Team Dispatch Triggered</div>
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
# CAMERA GRID (STATIC CAMERAS RENDER ONCE, ONLY CAM 01 IS FRAGMENT-SCOPED)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Active Monitoring Grid</div>', unsafe_allow_html=True)

armed = st.session_state.system_armed
live_badge  = '<span class="badge-live"><span class="dot dot-r" style="width:7px;height:7px;"></span>LIVE</span>'
stdby_badge = '<span class="badge-stdby">NO SIGNAL</span>'

def render_cam_01_feed(stream_url):
    """Dynamic feed, metric strip, silence button, and hazard detection for CAM 01."""
    if st.session_state.pop("silence_toast_pending", False):
        st.toast(f"🔕 CAM 01 alerts silenced for {SILENCE_DURATION_SEC}s.")

    armed = st.session_state.get("system_armed", False)

    now = time.time()
    is_silenced = (now < st.session_state.get('silenced_until', 0.0))

    # When disarmed, ALWAYS render metric strip and standby placeholder before returning
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

        st.markdown(
            '<div class="mstrip"><span class="ml">Crowd Count</span>'
            '<span class="mv off">—</span></div>',
            unsafe_allow_html=True
        )
        st.image(create_placeholder("CAM 01", "STANDBY"), width="stretch")
        st.button("🔕 Silence CAM 01", key="silence_1",
                  on_click=silence_alerts, width="stretch")
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

    # Read latest inference frame and values from worker
    with worker.lock:
        frame = worker.latest_frame
        person_count = worker.person_count
        fire_spotted = worker.fire_spotted

    st.session_state.live_crowd_count = person_count

    if frame is not None:
        st.markdown(
            f'<div class="mstrip"><span class="ml">Crowd Count</span>'
            f'<span class="mv">👥 {person_count}</span></div>',
            unsafe_allow_html=True
        )
        st.image(frame, channels="RGB", width="stretch")
    else:
        st.markdown(
            '<div class="mstrip"><span class="ml">Crowd Count</span>'
            '<span class="mv off">—</span></div>',
            unsafe_allow_html=True
        )
        st.image(create_placeholder("CAM 01", "CONNECTING..."), width="stretch")
        if str(stream_url).startswith("http"):
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:11px;color:var(--warn);text-align:center;padding:4px 0;">'
                f'🟡 AWAITING FEED ({stream_url}) &middot; Ensure "Start server" is active in IP Webcam on phone</div>',
                unsafe_allow_html=True
            )

    st.button("🔕 Silence CAM 01", key="silence_1",
              on_click=silence_alerts, width="stretch")

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
        return

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

    # Persistent Parent-Window Web Audio Playback (Loops infinitely until Silence/Disarm)
    active_b64 = st.session_state.get('active_audio_b64')
    active_id = st.session_state.get('active_audio_id', 0)

    if armed and not is_silenced and active_b64:
        play_persistent_audio(active_b64, active_id)
    else:
        if not armed or is_silenced:
            st.session_state.active_audio_b64 = None
            play_persistent_audio(None, 0)

col1, col2 = st.columns(2)

with col1:
    worker_obj = st.session_state.get('camera_worker')
    has_active_feed = (armed and worker_obj is not None and getattr(worker_obj, 'latest_frame', None) is not None)
    hdr_cls = "cam-hdr live" if has_active_feed else "cam-hdr"
    if armed:
        badge1 = live_badge if has_active_feed else '<span class="badge-live" style="background:#F2B33D;color:#0B0F14;">CONNECTING</span>'
    else:
        badge1 = stdby_badge
    st.markdown(f"""
    <div class="{hdr_cls}">
      <div>
        <div class="cam-nm">CAM 01</div>
        <div class="cam-loc">MIET College Main Gate</div>
      </div>
      {badge1}
    </div>""", unsafe_allow_html=True)

    if hasattr(st, "fragment"):
        @st.fragment(run_every=0.5)
        def cam_01_live_fragment(stream_url):
            render_cam_01_feed(stream_url)

        cam_01_live_fragment(ip_url)
    else:
        render_cam_01_feed(ip_url)
        if st.session_state.system_armed:
            time.sleep(0.5)
            st.rerun()

with col2:
    st.markdown(f"""
    <div class="cam-hdr">
      <div>
        <div class="cam-nm">CAM 02</div>
        <div class="cam-loc">Railway Station Entrance</div>
      </div>
      {stdby_badge}
    </div>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="mstrip"><span class="ml">Crowd Count</span>'
        '<span class="mv off">—</span></div>', unsafe_allow_html=True)
    st.image(create_placeholder("CAM 02"), width="stretch")
    st.button("🔕 Silence CAM 02", key="silence_2",
              on_click=dummy_silence, args=("CAM 02",), width="stretch")

col3, col4 = st.columns(2)

with col3:
    st.markdown(f"""
    <div class="cam-hdr">
      <div>
        <div class="cam-nm">CAM 03</div>
        <div class="cam-loc">Metro Station Entrance</div>
      </div>
      {stdby_badge}
    </div>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="mstrip"><span class="ml">Crowd Count</span>'
        '<span class="mv off">—</span></div>', unsafe_allow_html=True)
    st.image(create_placeholder("CAM 03"), width="stretch")
    st.button("🔕 Silence CAM 03", key="silence_3",
              on_click=dummy_silence, args=("CAM 03",), width="stretch")

with col4:
    st.markdown(f"""
    <div class="cam-hdr">
      <div>
        <div class="cam-nm">CAM 04</div>
        <div class="cam-loc">Server Room Backend</div>
      </div>
      {stdby_badge}
    </div>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="mstrip"><span class="ml">Crowd Count</span>'
        '<span class="mv off">—</span></div>', unsafe_allow_html=True)
    st.image(create_placeholder("CAM 04"), width="stretch")
    st.button("🔕 Silence CAM 04", key="silence_4",
              on_click=dummy_silence, args=("CAM 04",), width="stretch")

if "dummy_silence_cam" in st.session_state and st.session_state.dummy_silence_cam:
    _dcam = st.session_state.pop("dummy_silence_cam")
    st.toast(f"ℹ️ {_dcam} is on STANDBY — no active alerts to silence.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER STATUS BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
  <span>⬡ HAZARDLENS COMMAND V2.0 · MIET CAMPUS, MEERUT</span>
  <span>TWIN YOLOV8 · EDGE-TTS · RESEND</span>
  <span><span class="online">●</span> ALL TIMES IST (UTC+05:30)</span>
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
