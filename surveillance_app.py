"""
AI-Based Intelligent Surveillance & Intrusion Detection System
Streamlit Frontend — Manmeet Singh | Roll No: 12301176
"""
 
import streamlit as st
import sqlite3
import cv2
import numpy as np
import tempfile
import os
import time
import datetime
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import io
from PIL import Image
 
os.makedirs("models", exist_ok=True)
 
# ── Auto-download model weights from Google Drive if not present ──
import requests
import os

os.makedirs("models", exist_ok=True)

# Force re-download (remove after one successful deploy)
for f in ["models/person_model.pt", "models/weapon_model.pt", "models/vehicle_model.pt"]:
    if os.path.exists(f):
        os.remove(f)

def download_from_drive(file_id, dest_path):
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"

    # Step 1 — get confirmation token (don't stream this one)
    response = session.get(url, params={"id": file_id})
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if not token:
        import re
        match = re.search(r'confirm=([0-9A-Za-z_\-]+)', response.text)
        if match:
            token = match.group(1)

    # Step 2 — fresh request with token to get actual file
    params = {"id": file_id}
    if token:
        params["confirm"] = token

    download_response = session.get(url, params=params, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in download_response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

if not os.path.exists("models/person_model.pt"):
    download_from_drive("1jX8WjR3HXIf3DQON_rInKgSc6aqa0MU2", "models/person_model.pt")

if not os.path.exists("models/weapon_model.pt"):
    download_from_drive("1zVU5GuPSFjl7w0Uy7KvCiqDufxhTa7PD", "models/weapon_model.pt")

if not os.path.exists("models/vehicle_model.pt"):
    download_from_drive("1GaPznM4NScxOv8XJQNXwgsvbwVyuj_Sd", "models/vehicle_model.pt")

for name, path in [("person", "models/person_model.pt"), 
                   ("weapon", "models/weapon_model.pt"), 
                   ("vehicle", "models/vehicle_model.pt")]:
    size = os.path.getsize(path) if os.path.exists(path) else 0
    with open(path, "rb") as f:
        header = f.read(20)
    print(f"{name}: {size} bytes | header: {header}")
                    
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL — AI Surveillance",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────
# CUSTOM CSS — DARK TACTICAL THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');
 
/* ── GLOBAL ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #050A0F;
    color: #C8D8E4;
}
 
/* ── HIDE STREAMLIT DEFAULTS ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
 
/* ── SCANLINE OVERLAY ── */
body::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 255, 136, 0.015) 2px,
        rgba(0, 255, 136, 0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}
 
/* ── HEADER BANNER ── */
.sentinel-header {
    background: linear-gradient(135deg, #0A1628 0%, #050A0F 50%, #0A0F1A 100%);
    border: 1px solid #00FF88;
    border-left: 4px solid #FF3333;
    padding: 20px 30px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.sentinel-header::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00FF88, transparent);
    animation: scanH 3s linear infinite;
}
@keyframes scanH {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.header-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    color: #00FF88;
    letter-spacing: 6px;
    text-shadow: 0 0 30px rgba(0,255,136,0.5);
    margin: 0;
}
.header-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #667788;
    letter-spacing: 3px;
    margin-top: 4px;
}
.header-status {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #FF3333;
    letter-spacing: 2px;
    animation: blink 1.5s step-end infinite;
}
@keyframes blink {
    50% { opacity: 0; }
}
 
/* ── STAT CARDS ── */
.stat-card {
    background: linear-gradient(145deg, #0D1B2A, #080F1A);
    border: 1px solid #1A3040;
    border-top: 2px solid #00FF88;
    padding: 16px 20px;
    border-radius: 2px;
    position: relative;
}
.stat-card.danger { border-top-color: #FF3333; }
.stat-card.warning { border-top-color: #FFAA00; }
.stat-card.info { border-top-color: #00AAFF; }
 
.stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 3px;
    color: #445566;
    text-transform: uppercase;
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00FF88;
    line-height: 1.1;
}
.stat-card.danger .stat-value { color: #FF3333; }
.stat-card.warning .stat-value { color: #FFAA00; }
.stat-card.info .stat-value { color: #00AAFF; }
 
/* ── SECTION HEADINGS ── */
.section-head {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    color: #00FF88;
    border-bottom: 1px solid #0D2030;
    padding-bottom: 8px;
    margin-bottom: 16px;
    text-transform: uppercase;
}
 
/* ── ALERT BOXES ── */
.alert-critical {
    background: rgba(255,51,51,0.1);
    border: 1px solid #FF3333;
    border-left: 4px solid #FF3333;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: #FF8888;
    animation: pulse-red 2s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { border-left-color: #FF3333; }
    50% { border-left-color: #FF8888; }
}
.alert-warning {
    background: rgba(255,170,0,0.08);
    border: 1px solid #FFAA00;
    border-left: 4px solid #FFAA00;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: #FFCC66;
}
.alert-clear {
    background: rgba(0,255,136,0.06);
    border: 1px solid #00FF88;
    border-left: 4px solid #00FF88;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: #66FFAA;
}
 
/* ── TABLE STYLING ── */
.dataframe thead tr th {
    background: #0A1628 !important;
    color: #00FF88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    border-bottom: 1px solid #1A3040 !important;
}
.dataframe tbody tr { background: #080F1A !important; }
.dataframe tbody tr:hover { background: #0D1B2A !important; }
.dataframe tbody tr td {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #A0B8C8 !important;
    border-color: #0D2030 !important;
}
 
/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #060C12 !important;
    border-right: 1px solid #0D2030;
}
[data-testid="stSidebar"] .css-1d391kg { background: #060C12 !important; }
 
/* ── BUTTONS ── */
.stButton>button {
    background: transparent !important;
    border: 1px solid #00FF88 !important;
    color: #00FF88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    padding: 8px 20px !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
.stButton>button:hover {
    background: rgba(0,255,136,0.1) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
}
 
/* ── INPUTS ── */
.stSlider, .stSelectbox, .stFileUploader {
    font-family: 'Rajdhani', sans-serif !important;
}
 
/* ── VIDEO CONTAINER ── */
.video-feed {
    border: 1px solid #1A3040;
    border-top: 2px solid #00AAFF;
    background: #020608;
    padding: 4px;
    position: relative;
}
.feed-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: #00AAFF;
    letter-spacing: 3px;
    padding: 4px 8px;
    background: rgba(0,170,255,0.1);
    display: inline-block;
    margin-bottom: 6px;
}
 
/* ── THREAT BADGE ── */
.threat-armed {
    background: #FF3333;
    color: white;
    padding: 2px 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    font-weight: bold;
}
.threat-unarmed {
    background: #00AA66;
    color: #001A0D;
    padding: 2px 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
}
.threat-vehicle {
    background: #0055AA;
    color: white;
    padding: 2px 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
}
 
/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #050A0F; }
::-webkit-scrollbar-thumb { background: #1A3040; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #00FF88; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────
DB_PATH = "surveillance_logs.db"
 
def init_db():
    """Initialize SQLite database with all required tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
 
    # Detection Events Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS detection_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            source_file TEXT    NOT NULL,
            object_type TEXT    NOT NULL,
            threat_level TEXT   NOT NULL,
            confidence  REAL    NOT NULL,
            bbox_x1     INTEGER,
            bbox_y1     INTEGER,
            bbox_x2     INTEGER,
            bbox_y2     INTEGER,
            frame_no    INTEGER,
            session_id  TEXT
        )
    """)
 
    # Session Summary Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    UNIQUE NOT NULL,
            started_at      TEXT    NOT NULL,
            ended_at        TEXT,
            source_file     TEXT,
            total_frames    INTEGER DEFAULT 0,
            armed_count     INTEGER DEFAULT 0,
            unarmed_count   INTEGER DEFAULT 0,
            vehicle_count   INTEGER DEFAULT 0,
            weapon_count    INTEGER DEFAULT 0
        )
    """)
 
    conn.commit()
    conn.close()
 
 
def log_detection(session_id, source_file, object_type, threat_level, confidence,
                  bbox=None, frame_no=None):
    """Insert a detection event into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (bbox if bbox else (None, None, None, None))
    c.execute("""
        INSERT INTO detection_events
        (timestamp, source_file, object_type, threat_level, confidence,
         bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_no, session_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (ts, source_file, object_type, threat_level, confidence,
          bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_no, session_id))
    conn.commit()
    conn.close()
 
 
def create_session(session_id, source_file):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT OR IGNORE INTO sessions (session_id, started_at, source_file)
        VALUES (?,?,?)
    """, (session_id, ts, source_file))
    conn.commit()
    conn.close()
 
 
def close_session(session_id, total_frames):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    # Count per type
    c.execute("SELECT object_type, COUNT(*) FROM detection_events WHERE session_id=? GROUP BY object_type",
              (session_id,))
    counts = dict(c.fetchall())
 
    c.execute("""
        UPDATE sessions SET ended_at=?, total_frames=?,
        armed_count=?, unarmed_count=?, vehicle_count=?, weapon_count=?
        WHERE session_id=?
    """, (ts, total_frames,
          counts.get("Armed Person", 0),
          counts.get("Unarmed Person", 0),
          counts.get("Vehicle", 0),
          counts.get("Weapon", 0),
          session_id))
    conn.commit()
    conn.close()
 
 
def get_recent_logs(limit=100):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT timestamp, object_type, threat_level, confidence,
               source_file, frame_no, session_id
        FROM detection_events
        ORDER BY id DESC LIMIT {limit}
    """, conn)
    conn.close()
    return df
 
 
def get_session_summary():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT session_id, started_at, ended_at, source_file,
               total_frames, armed_count, unarmed_count, vehicle_count, weapon_count
        FROM sessions ORDER BY id DESC LIMIT 20
    """, conn)
    conn.close()
    return df
 
 
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM detection_events")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM detection_events WHERE threat_level='CRITICAL'")
    critical = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM detection_events WHERE threat_level='WARNING'")
    warning = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM sessions")
    sessions = c.fetchone()[0]
    conn.close()
    return total, critical, warning, sessions
 
 
def get_threat_timeline():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT DATE(timestamp) as date, threat_level, COUNT(*) as count
        FROM detection_events
        GROUP BY date, threat_level
        ORDER BY date DESC LIMIT 60
    """, conn)
    conn.close()
    return df
 
 
# ─────────────────────────────────────────────
# MODEL LOADING (CACHED)
# ─────────────────────────────────────────────
 
from ultralytics import YOLO
 
def load_model_safe(path):
    try:
        # Try normal loading
        return YOLO(path)
    except Exception as e:
        print(f"⚠ Normal load failed for {path}, using safe load...")
 
        # Safe fallback (fixes DFLoss error)
        model = YOLO("yolov8s.pt")
        model.load(path)
        return model
    
@st.cache_resource
def load_models(person_path, weapon_path, vehicle_path):
    models = {}
 
    if person_path and os.path.exists(person_path):
        models["person"] = load_model_safe(person_path)
 
    if weapon_path and os.path.exists(weapon_path):
        models["weapon"] = load_model_safe(weapon_path)
 
    if vehicle_path and os.path.exists(vehicle_path):
        models["vehicle"] = load_model_safe(vehicle_path)
 
    return models
 
 
# ─────────────────────────────────────────────
# DETECTION CORE
# ─────────────────────────────────────────────
def iou_overlap(b1, b2):
    """Returns True if two bboxes overlap (used for armed-person logic)."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    return ix1 < ix2 and iy1 < iy2
 
 
def run_detection_frame(frame, models, conf_thresh, session_id, source_name, frame_no):
    """
    Run all loaded models on a single frame.
    Returns (annotated_frame, detections_list)
    detections = [{"object_type", "threat_level", "confidence", "bbox"}, ...]
    """
    annotated = frame.copy()
    detections = []
 
    persons = []
    weapons = []
 
    # ── PERSON DETECTION ──
    if "person" in models:
        p_res = models["person"].track(frame, persist=True, conf=conf_thresh, iou=0.5, verbose=False)
        if p_res[0].boxes is not None:
            for box, conf in zip(p_res[0].boxes.xyxy, p_res[0].boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                c = float(conf)
                persons.append((x1, y1, x2, y2, c))
                # Default: unarmed (will be overridden below if weapon overlaps)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 100), 2)
                cv2.putText(annotated, f"Person {c:.0%}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 2)
 
    # ── WEAPON DETECTION ──
    if "weapon" in models:
        w_res = models["weapon"].predict(frame, conf=conf_thresh, iou=0.5, verbose=False)
        if w_res[0].boxes is not None:
            for box, conf in zip(w_res[0].boxes.xyxy, w_res[0].boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                c = float(conf)
                weapons.append((x1, y1, x2, y2, c))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 100, 0), 2)
                cv2.putText(annotated, f"Weapon {c:.0%}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 100, 0), 2)
                det = {"object_type": "Weapon", "threat_level": "WARNING",
                       "confidence": c, "bbox": (x1, y1, x2, y2)}
                detections.append(det)
 
    # ── VEHICLE DETECTION ──
    if "vehicle" in models:
        v_res = models["vehicle"].predict(frame, conf=conf_thresh, iou=0.5, verbose=False)
        if v_res[0].boxes is not None:
            for box, conf in zip(v_res[0].boxes.xyxy, v_res[0].boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                c = float(conf)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 120, 255), 2)
                cv2.putText(annotated, f"Vehicle {c:.0%}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)
                det = {"object_type": "Vehicle", "threat_level": "LOW",
                       "confidence": c, "bbox": (x1, y1, x2, y2)}
                detections.append(det)
 
    # ── ARMED / UNARMED CLASSIFICATION ──
    armed_person_ids = set()
    for i, (px1, py1, px2, py2, pc) in enumerate(persons):
        is_armed = False
        for (wx1, wy1, wx2, wy2, wc) in weapons:
            if iou_overlap((px1, py1, px2, py2), (wx1, wy1, wx2, wy2)):
                is_armed = True
                break
 
        if is_armed:
            # Redraw as ARMED
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)
            cv2.putText(annotated, "ARMED!", (px1, py1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            det = {"object_type": "Armed Person", "threat_level": "CRITICAL",
                   "confidence": pc, "bbox": (px1, py1, px2, py2)}
            armed_person_ids.add(i)
        else:
            det = {"object_type": "Unarmed Person", "threat_level": "LOW",
                   "confidence": pc, "bbox": (px1, py1, px2, py2)}
        detections.append(det)
 
    # ── TIMESTAMP OVERLAY ──
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(annotated, f"SENTINEL  |  {ts}", (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 80), 1)
    cv2.putText(annotated, f"FRAME {frame_no:06d}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 80), 1)
 
    # ── LOG TO DB ──
    for det in detections:
        log_detection(session_id, source_name,
                      det["object_type"], det["threat_level"],
                      det["confidence"], det["bbox"], frame_no)
 
    return annotated, detections
 
 
# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
init_db()
 
# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "processing" not in st.session_state:
    st.session_state.processing = False
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
 
# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="sentinel-header">
    <div class="header-title">⬡ SENTINEL</div>
    <div class="header-subtitle">AI-BASED INTELLIGENT SURVEILLANCE & INTRUSION DETECTION SYSTEM</div>
    <div class="header-status">● SYSTEM ACTIVE — GROUND LEVEL MONITORING</div>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-head">⚙ SYSTEM CONFIGURATION</div>', unsafe_allow_html=True)
 
    st.markdown("**MODEL PATHS**")
    person_model_path = st.text_input(
        "Person Model (.pt)", value="models/person_model.pt",
        help="Path to your trained person detection YOLO model"
    )
    weapon_model_path = st.text_input(
        "Weapon Model (.pt)", value="models/weapon_model.pt",
        help="Path to your trained weapon detection YOLO model"
    )
    vehicle_model_path = st.text_input(
        "Vehicle Model (.pt)", value="models/vehicle_model.pt",
        help="Path to your trained vehicle detection YOLO model"
    )
 
    st.markdown("---")
    st.markdown("**DETECTION SETTINGS**")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.40, 0.05)
    process_every_n = st.slider("Process Every N Frames", 1, 10, 2,
                                help="Skip frames for faster processing")
    show_all_detections = st.checkbox("Log ALL detections (not just threats)", value=True)
 
    st.markdown("---")
    st.markdown("**DISPLAY SETTINGS**")
    show_feed = st.checkbox("Show Live Feed", value=True)
    feed_width = st.slider("Feed Width (px)", 400, 1200, 700)
 
    st.markdown("---")
    st.markdown('<div class="section-head">📊 QUICK STATS</div>', unsafe_allow_html=True)
    total_ev, crit_ev, warn_ev, num_sess = get_stats()
    st.markdown(f"""
    <div class="stat-card danger" style="margin-bottom:8px">
        <div class="stat-label">CRITICAL ALERTS</div>
        <div class="stat-value">{crit_ev}</div>
    </div>
    <div class="stat-card warning" style="margin-bottom:8px">
        <div class="stat-label">TOTAL EVENTS</div>
        <div class="stat-value">{total_ev}</div>
    </div>
    <div class="stat-card info" style="margin-bottom:8px">
        <div class="stat-label">SESSIONS RUN</div>
        <div class="stat-value">{num_sess}</div>
    </div>
    """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  DETECTION", "📋  EVENT LOG", "📈  ANALYTICS", "ℹ  ABOUT"
])
 
 
# ══════════════════════════════════════════════
# TAB 1 — DETECTION
# ══════════════════════════════════════════════
with tab1:
    col_feed, col_alerts = st.columns([3, 1])
 
    with col_feed:
        st.markdown('<div class="section-head">▶ VIDEO INPUT</div>', unsafe_allow_html=True)
 
        input_mode = st.radio(
            "Input Source",
            ["Upload Video File", "Use Webcam (Live)"],
            horizontal=True,
            label_visibility="collapsed"
        )
 
        uploaded_video = None
        use_webcam = False
 
        if input_mode == "Upload Video File":
            uploaded_video = st.file_uploader(
                "Upload surveillance footage",
                type=["mp4", "avi", "mov", "mkv"],
                label_visibility="collapsed"
            )
        else:
            use_webcam = True
            st.info("⚡ Webcam mode — click START to begin live detection")
 
        # ── CONTROLS ──
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        with ctrl_col1:
            start_btn = st.button("▶  START SCAN", use_container_width=True)
        with ctrl_col2:
            stop_btn  = st.button("■  STOP", use_container_width=True)
        with ctrl_col3:
            clear_btn = st.button("⟳  CLEAR ALERTS", use_container_width=True)
 
        if stop_btn:
            st.session_state.stop_flag = True
        if clear_btn:
            st.session_state.alert_log = []
 
        # ── VIDEO PLACEHOLDER ──
        feed_placeholder   = st.empty()
        status_placeholder = st.empty()
        progress_bar       = st.empty()
 
    with col_alerts:
        st.markdown('<div class="section-head">🚨 LIVE ALERTS</div>', unsafe_allow_html=True)
        alerts_placeholder = st.empty()
 
    # ── RENDER IDLE FEED ──
    if not start_btn:
        feed_placeholder.markdown("""
        <div class="video-feed" style="height:380px; display:flex; align-items:center; justify-content:center;">
            <div style="text-align:center;">
                <div style="font-family:'Orbitron',monospace; font-size:2rem; color:#1A3040;">◉</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            color:#334455; letter-spacing:3px; margin-top:8px;">
                    AWAITING INPUT
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        alerts_placeholder.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.6rem; '
            'color:#334455; letter-spacing:2px;">NO ACTIVE SESSION</div>',
            unsafe_allow_html=True
        )
 
    # ── START DETECTION ──
    if start_btn and (uploaded_video is not None or use_webcam):
        st.session_state.stop_flag = False
 
        # Load models
        with st.spinner("Loading AI models..."):
            models = load_models(person_model_path, weapon_model_path, vehicle_model_path)
 
        if not models:
            st.error("⚠ No models loaded. Check your model paths in the sidebar.")
        else:
            # Session
            session_id = datetime.datetime.now().strftime("SES_%Y%m%d_%H%M%S")
            source_name = uploaded_video.name if uploaded_video else "WEBCAM"
            create_session(session_id, source_name)
 
            # Write video to temp file
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.flush()
                video_path = tfile.name
            else:
                video_path = 0  # webcam
 
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            frame_no = 0
            session_detections = []
 
            while cap.isOpened() and not st.session_state.stop_flag:
                ret, frame = cap.read()
                if not ret:
                    break
 
                frame_no += 1
 
                if frame_no % process_every_n != 0:
                    continue
 
                # Run detection
                annotated, detections = run_detection_frame(
                    frame, models, conf_thresh, session_id, source_name, frame_no
                )
                session_detections.extend(detections)
 
                # Show feed
                if show_feed:
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    feed_placeholder.image(img, width=feed_width,
                                           caption=f"SENTINEL FEED — Frame {frame_no}")
 
                # Update alerts panel
                critical_now = [d for d in detections if d["threat_level"] == "CRITICAL"]
                for c in critical_now:
                    st.session_state.alert_log.insert(0, {
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "type": c["object_type"],
                        "level": c["threat_level"],
                        "conf": c["confidence"],
                    })
 
                # Render alerts
                if st.session_state.alert_log:
                    html = ""
                    for a in st.session_state.alert_log[:8]:
                        cls = "alert-critical" if a["level"] == "CRITICAL" else "alert-warning"
                        html += f"""
                        <div class="{cls}">
                            <b>{a['time']}</b><br>
                            {a['type']}<br>
                            CONF: {a['conf']:.0%}
                        </div>"""
                    alerts_placeholder.markdown(html, unsafe_allow_html=True)
                else:
                    alerts_placeholder.markdown(
                        '<div class="alert-clear">◉ ALL CLEAR — NO THREATS</div>',
                        unsafe_allow_html=True
                    )
 
                # Progress
                if total_frames > 0:
                    pct = frame_no / total_frames
                    progress_bar.progress(min(pct, 1.0),
                        text=f"Processing frame {frame_no}/{total_frames}")
 
                status_placeholder.markdown(
                    f'<div style="font-family:\'Share Tech Mono\',monospace; '
                    f'font-size:0.65rem; color:#00FF88; letter-spacing:2px;">'
                    f'SESSION: {session_id} | FRAME: {frame_no} | '
                    f'DETECTIONS: {len(session_detections)}</div>',
                    unsafe_allow_html=True
                )
 
            cap.release()
            if uploaded_video:
                os.unlink(video_path)
 
            close_session(session_id, frame_no)
            progress_bar.progress(1.0, text="✅ SCAN COMPLETE")
            status_placeholder.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; '
                f'color:#00FF88; letter-spacing:2px;">'
                f'SESSION COMPLETE: {frame_no} frames | {len(session_detections)} total detections</div>',
                unsafe_allow_html=True
            )
 
    elif start_btn:
        st.warning("Please upload a video file or select webcam mode.")
 
 
# ══════════════════════════════════════════════
# TAB 2 — EVENT LOG
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">📋 DETECTION EVENT LOG</div>', unsafe_allow_html=True)
 
    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        filter_threat = st.selectbox("Filter by Threat Level",
                                     ["ALL", "CRITICAL", "WARNING", "LOW"])
    with f2:
        filter_type = st.selectbox("Filter by Object Type",
                                   ["ALL", "Armed Person", "Unarmed Person", "Vehicle", "Weapon"])
    with f3:
        log_limit = st.slider("Show last N events", 10, 500, 100)
 
    col_refresh, col_export = st.columns([1, 5])
    with col_refresh:
        if st.button("⟳  REFRESH"):
            st.rerun()
 
    df_logs = get_recent_logs(log_limit)
 
    if df_logs.empty:
        st.markdown('<div class="alert-clear">◉ NO EVENTS IN DATABASE — run a detection scan first.</div>',
                    unsafe_allow_html=True)
    else:
        # Apply filters
        if filter_threat != "ALL":
            df_logs = df_logs[df_logs["threat_level"] == filter_threat]
        if filter_type != "ALL":
            df_logs = df_logs[df_logs["object_type"] == filter_type]
 
        # Color-coded metrics
        m1, m2, m3, m4 = st.columns(4)
        all_logs = get_recent_logs(10000)
        with m1:
            st.markdown(f"""
            <div class="stat-card danger">
                <div class="stat-label">ARMED PERSONS</div>
                <div class="stat-value">{len(all_logs[all_logs['object_type']=='Armed Person'])}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="stat-card warning">
                <div class="stat-label">WEAPONS</div>
                <div class="stat-value">{len(all_logs[all_logs['object_type']=='Weapon'])}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">UNARMED PERSONS</div>
                <div class="stat-value">{len(all_logs[all_logs['object_type']=='Unarmed Person'])}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="stat-card info">
                <div class="stat-label">VEHICLES</div>
                <div class="stat-value">{len(all_logs[all_logs['object_type']=='Vehicle'])}</div>
            </div>""", unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_logs, use_container_width=True, height=400)
 
        # Export
        csv_data = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  EXPORT LOG AS CSV",
            data=csv_data,
            file_name=f"sentinel_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
 
    st.markdown("---")
    st.markdown('<div class="section-head">🗃 SESSION HISTORY</div>', unsafe_allow_html=True)
    df_sess = get_session_summary()
    if not df_sess.empty:
        st.dataframe(df_sess, use_container_width=True, height=250)
    else:
        st.markdown('<div class="alert-clear">◉ NO SESSIONS RECORDED YET.</div>',
                    unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">📈 ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)
 
    df_all = get_recent_logs(10000)
 
    if df_all.empty:
        st.markdown('<div class="alert-clear">◉ NO DATA — run detection scans to populate analytics.</div>',
                    unsafe_allow_html=True)
    else:
        # ── ROW 1: Distribution charts ──
        c1, c2 = st.columns(2)
 
        with c1:
            st.markdown("**DETECTIONS BY OBJECT TYPE**")
            type_counts = df_all["object_type"].value_counts().reset_index()
            type_counts.columns = ["Object Type", "Count"]
            st.bar_chart(type_counts.set_index("Object Type"), color="#00FF88")
 
        with c2:
            st.markdown("**DETECTIONS BY THREAT LEVEL**")
            threat_counts = df_all["threat_level"].value_counts().reset_index()
            threat_counts.columns = ["Threat Level", "Count"]
            colors = {"CRITICAL": "#FF3333", "WARNING": "#FFAA00", "LOW": "#00FF88"}
            st.bar_chart(threat_counts.set_index("Threat Level"))
 
        # ── ROW 2: Confidence distribution ──
        st.markdown("**CONFIDENCE SCORE DISTRIBUTION**")
        st.bar_chart(df_all["confidence"].round(1).value_counts().sort_index())
 
        # ── ROW 3: Timeline ──
        if "timestamp" in df_all.columns and len(df_all) > 1:
            st.markdown("**DETECTION TIMELINE (events over time)**")
            df_all["hour"] = pd.to_datetime(df_all["timestamp"]).dt.floor("H")
            timeline = df_all.groupby(["hour", "threat_level"]).size().reset_index(name="count")
            if not timeline.empty:
                pivot = timeline.pivot(index="hour", columns="threat_level", values="count").fillna(0)
                st.line_chart(pivot)
 
        # ── ROW 4: Source file breakdown ──
        st.markdown("**DETECTIONS BY SOURCE FILE**")
        src_counts = df_all["source_file"].value_counts().head(10).reset_index()
        src_counts.columns = ["Source", "Count"]
        st.bar_chart(src_counts.set_index("Source"), color="#00AAFF")
 
        # ── Session summary ──
        df_s = get_session_summary()
        if not df_s.empty:
            st.markdown("**SESSION SUMMARY TABLE**")
            st.dataframe(df_s, use_container_width=True)
 
 
# ══════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="max-width:800px;">
    <div class="section-head">ℹ SYSTEM INFORMATION</div>
 
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                color:#7A9BAC; line-height:2; letter-spacing:1px;">
    <p>
    <span style="color:#00FF88;">PROJECT</span>&nbsp;&nbsp;: AI-Based Intelligent Surveillance &amp;
    Intrusion Detection System<br>
    <span style="color:#00FF88;">STUDENT</span>&nbsp;&nbsp;: Manmeet Singh | Roll No: 12301176<br>
    <span style="color:#00FF88;">DEGREE</span>&nbsp;&nbsp;&nbsp;: B.Tech CSE (2023–27), Semester 6 — Group 5<br>
    <span style="color:#00FF88;">DEPT</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Department of Computer Science &amp; Engineering
    </p>
 
    <div class="section-head" style="margin-top:24px;">🔺 DETECTION CAPABILITIES</div>
 
    <p>
    <span style="color:#FFAA00;">MODEL 1 — VEHICLE DETECTOR</span><br>
    &nbsp;&nbsp;• Trained on VisDrone + Vehicle Detection 2.0 datasets<br>
    &nbsp;&nbsp;• Single class: "vehicle"<br>
    &nbsp;&nbsp;• YOLOv8 custom trained, ~110+ epochs<br><br>
 
    <span style="color:#FFAA00;">MODEL 2 — WEAPON DETECTOR</span><br>
    &nbsp;&nbsp;• Trained on Armed Person Recognition dataset (Roboflow)<br>
    &nbsp;&nbsp;• Detects firearms &amp; weapons in surveillance footage<br>
    &nbsp;&nbsp;• YOLOv8s, 60 epochs<br><br>
 
    <span style="color:#FFAA00;">MODEL 3 — PERSON DETECTOR</span><br>
    &nbsp;&nbsp;• Trained on merged Person + CrowdHuman datasets<br>
    &nbsp;&nbsp;• Single class: "person" with tracking (ByteTrack)<br>
    &nbsp;&nbsp;• YOLOv8s, multi-stage training<br><br>
 
    <span style="color:#FF3333;">ARMED / UNARMED CLASSIFICATION</span><br>
    &nbsp;&nbsp;• Person model + Weapon model combined via IoU overlap logic<br>
    &nbsp;&nbsp;• If weapon bbox overlaps person bbox → ARMED PERSON (CRITICAL alert)<br>
    &nbsp;&nbsp;• Otherwise → UNARMED PERSON (LOW threat)
    </p>
 
    <div class="section-head" style="margin-top:24px;">🗄 DATABASE SCHEMA</div>
 
    <p>
    SQLite database: <span style="color:#00FF88;">surveillance_logs.db</span><br><br>
    <span style="color:#00AAFF;">detection_events</span> — every individual detection event<br>
    &nbsp;&nbsp;Fields: timestamp, source_file, object_type, threat_level,<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;confidence, bbox coords, frame_no, session_id<br><br>
    <span style="color:#00AAFF;">sessions</span> — one row per detection run<br>
    &nbsp;&nbsp;Fields: session_id, started_at, ended_at, source_file,<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;total_frames, armed_count, unarmed_count, vehicle_count
    </p>
 
    <div class="section-head" style="margin-top:24px;">⚙ TECH STACK</div>
 
    <p>
    Python &nbsp;|&nbsp; YOLOv8 (Ultralytics) &nbsp;|&nbsp; OpenCV &nbsp;|&nbsp;
    Streamlit &nbsp;|&nbsp; SQLite &nbsp;|&nbsp; Pandas
    </p>
    </div>
    </div>
    """, unsafe_allow_html=True)
