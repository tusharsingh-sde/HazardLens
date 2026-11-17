<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-27338E?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Edge--TTS-0078D4?style=for-the-badge&logo=microsoft&logoColor=white" alt="Edge-TTS" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<h1 align="center">⬡ HazardLens</h1>

<p align="center">
  <strong>AI-Powered Smart City Surveillance Command Center</strong><br/>
  <em>Real-Time Fire Detection · Crowd Monitoring · Hindi Neural Voice Alerts · QR Camera Pairing · Email Dispatch</em>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#️-system-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#️-camera-pairing">Camera Pairing</a> •
  <a href="#️-configuration">Configuration</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-license">License</a>
</p>

---

## 📖 Overview

**HazardLens** is an edge-optimized, autonomous AI surveillance system that transforms a smartphone or IP camera into an intelligent hazard monitoring agent. It runs a **dual YOLOv8 inference pipeline** — a custom-trained fire/smoke model alongside a general-purpose crowd counter — to continuously analyse live video feeds and trigger multi-channel emergency alerts with **zero cloud dependency**.

Camera pairing is **completely wireless** — scan a QR code from your phone or click **⚡ Auto-Link** and the system discovers, verifies, and streams from your phone's IP Webcam app in seconds.

When a hazard is detected, the system simultaneously:

- Renders an animated tactical alert banner on the command center dashboard
- **Loops a Hindi neural voice alert infinitely** via Microsoft Edge-TTS (`hi-IN-SwaraNeural`) until the operator actively silences it
- Dispatches an **automated HTML email** to the configured security administrator
- Applies a **"Latest Hazard Wins"** state machine to prevent alert stacking, UI freezes, and audio collisions

---

## ❓ Problem Statement

Traditional CCTV systems are **passive monitoring tools**. In emergency situations, incidents are detected only after a human operator notices them, verifies them, and manually escalates the alarm. This critical delay increases physical damage, risks human life, and slows down emergency response.

**HazardLens** bridges this gap by providing **low-latency automated hazard detection, smart alert prioritization, instant localized voice broadcasts, and zero-configuration wireless camera pairing** — all running at the edge without any cloud servers.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔥 **Fire & Smoke Detection** | Custom fine-tuned YOLOv8 model with multilingual class mapping (EN/ZH) and `conf ≥ 0.80` threshold |
| 👥 **Crowd Monitoring** | Real-time person counting via YOLOv8n with configurable overcrowding threshold |
| 🎙️ **Infinite Hindi Voice Alerts** | Distinct Hindi voice alerts for Fire and Overcrowding via Edge-TTS `hi-IN-SwaraNeural` — loop until operator presses 🔕 Silence |
| 📲 **QR Camera Pairing** | WhatsApp-style zero-input wireless camera pairing — scan QR code on phone or click ⚡ Auto-Link |
| 🔗 **Dynamic IP Auto-Heal** | Camera IP is re-discovered automatically on every new Wi-Fi network; `config/paired_cameras.json` stores the last known IP |
| 📡 **HTTP Frame Grabber** | Fetches `/shot.jpg` snapshots from IP Webcam via `urllib` — bypasses OpenCV FFmpeg MJPEG bugs with zero buffer lag |
| 📧 **Automated Email Dispatch** | Queued HTML email alerts sent to the security admin via Resend API (non-blocking background worker with retry) |
| ⚡ **"Latest Hazard Wins" Engine** | Prevents UI freezes, alert looping, and audio overlapping via a finite state machine |
| 🖥️ **Tactical Command Center UI** | Dark-themed military-inspired dashboard with 4-camera grid, live metrics, and animated alert banners |
| 🎯 **False-Positive Immunity** | Optimized detection confidence thresholds and 3s debounce grace window to prevent flickering on borderline detections |
| 🔐 **Operator PIN Gate** | Timing-safe PIN authentication protects ARM / DISARM / SILENCE controls |
| 🛡️ **SSRF Protection** | URL validator blocks private metadata endpoints, loopback bypasses, and non-HTTP/RTSP schemes |
| 🐳 **Docker Support** | One-command production deployment with Docker Compose |

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────┐
                    │  📱 Smartphone (IP Webcam app) │
                    │   OR any RTSP / IP Camera     │
                    └─────────────┬────────────────┘
                                  │  Wi-Fi / LAN
                    ┌─────────────▼────────────────┐
                    │  QR Pairing / Auto-Link       │
                    │  (utils/qr_pairing.py)        │
                    │  Dynamic IP Registry          │
                    │  (config/paired_cameras.json) │
                    └─────────────┬────────────────┘
                                  │
                    ┌─────────────▼────────────────┐
                    │  HTTP Frame Grabber           │
                    │  GET /shot.jpg  (urllib)      │
                    │  Fallback: cv2.VideoCapture   │
                    └─────────────┬────────────────┘
                                  │
                    ┌─────────────┴────────────┐
                    │                          │
                    ▼                          ▼
          ┌──────────────────┐      ┌─────────────────┐
          │  🔥 Engine 1      │      │  👥 Engine 2    │
          │  Fire / Smoke     │      │  Crowd Counter  │
          │  (best.pt ~6MB)   │      │  (YOLOv8n ~6MB) │
          │  conf ≥ 0.80      │      │  class: person  │
          └────────┬─────────┘      └────────┬────────┘
                   │                          │
                   └────────────┬─────────────┘
                                │
                   ┌────────────▼─────────────┐
                   │  Hazard Detection Engine  │
                   │  "Latest Hazard Wins" FSM │
                   │  3s Debounce Grace Window │
                   └────────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
    ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ 🖥️ Visual    │  │ 🎙️ Hindi Voice   │  │ 📧 Email Alert   │
    │ Alert Banner  │  │ Alert (looping)  │  │ (Resend Worker)  │
    │ (Streamlit)   │  │ fire_alert.mp3   │  │ (Background)     │
    │               │  │ crowd_alert.mp3  │  │                  │
    └──────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend & Dashboard** | [Streamlit](https://streamlit.io/) ≥ 1.30 with custom CSS dark command-center theme |
| **Computer Vision** | [OpenCV](https://opencv.org/) (`opencv-python-headless`) |
| **Object Detection** | [Ultralytics YOLOv8](https://docs.ultralytics.com/) — dual model inference |
| **Fire/Smoke Model** | Custom fine-tuned YOLOv8 (`best.pt` — ~6 MB, included in repo) |
| **Crowd Model** | YOLOv8n pre-trained (`yolov8n.pt` — ~6 MB, auto-downloaded) |
| **Voice Engine** | [Microsoft Edge-TTS](https://pypi.org/project/edge-tts/) ≥ 7.3.1 (`hi-IN-SwaraNeural`) |
| **Camera Pairing** | `utils/qr_pairing.py` — QR code generation, LAN scanner, dynamic IP registry |
| **Stream Ingestion** | HTTP `/shot.jpg` frame grabber (urllib) + OpenCV VideoCapture fallback |
| **Email Alerts** | [Resend](https://resend.com) Python SDK (queued background worker with retry) |
| **QR Code** | `qrcode` library (pure-Python PNG generation) |
| **Runtime** | Python 3.10+ with `asyncio` |
| **Containerization** | Docker + Docker Compose |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10 or higher** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)
- **pip** (comes bundled with Python 3.10+)
- A phone running **[IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)** (Android) or any RTSP/HTTP IP camera on the same Wi-Fi

---

### 1 · Clone the Repository

```bash
git clone https://github.com/tusharsingh-sde/HazardLens.git
cd HazardLens
```

---

### 2 · Create a Virtual Environment

<details>
<summary><strong>🪟 Windows (PowerShell)</strong></summary>

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Note:** If you encounter an execution policy error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

</details>

<details>
<summary><strong>🪟 Windows (Command Prompt)</strong></summary>

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

</details>

<details>
<summary><strong>🐧 Linux (Ubuntu / Debian)</strong></summary>

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note:** If `python3-venv` is not installed:
> ```bash
> sudo apt update && sudo apt install python3-venv -y
> ```

</details>

<details>
<summary><strong>🍎 macOS</strong></summary>

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note:** If Python is not installed, use [Homebrew](https://brew.sh/):
> ```bash
> brew install python@3.10
> ```

</details>

---

### 3 · Install Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `streamlit >=1.30.0` | Dashboard framework |
| `ultralytics >=8.1.0` | YOLOv8 inference engine |
| `opencv-python-headless >=4.8.0` | Video frame processing |
| `edge-tts >=6.1.10` | Hindi neural voice synthesis |
| `numpy >=1.24.0` | Numerical array operations |
| `pillow >=10.0.0` | Image utilities |
| `python-dotenv >=1.0.0` | Environment configuration management |
| `resend >=2.0.0` | Official Resend API client for email alerts |
| `qrcode >=8.0.0` | QR code generation for camera pairing |
| `dill >=0.3.8` | Extended serialization support |

---

### 4 · Configure Environment

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and fill in your settings:

```env
# Resend API for Automated Email Alerts
RESEND_API_KEY=re_your_api_key_here
HAZARDLENS_FROM_EMAIL=onboarding@resend.dev
HAZARDLENS_ADMIN_EMAIL=admin-recipient@example.com

# Operator Passkey Gate (protects ARM / DISARM / SILENCE controls)
HAZARDLENS_OPERATOR_PIN=1234

# PyTorch Model Integrity Check (SHA-256 of best.pt)
HAZARDLENS_MODEL_SHA256=18d47a2e606035a2f1b5303050125a3c6c53b679a7a1b27b25bd72d7a63b3ec3
```

> **Security Note:** Never commit your `.env` file — it is listed in `.gitignore`.  
> **Tip:** Get a free API key at [resend.com](https://resend.com). For testing, `onboarding@resend.dev` can send to your verified account email.

---

### 5 · Run the Application

```bash
streamlit run app.py
```

The command center launches in your default browser at **`http://localhost:8501`**.

---

## 📲 Camera Pairing

HazardLens uses a **WhatsApp-style zero-input wireless pairing** system. Your phone becomes the CCTV camera — no cables, no manual IP configuration.

### Option A — QR Code Pairing

1. Open the **IP Webcam** app on your Android phone
2. Tap **"Start server"** at the bottom — the app displays an IP like `http://192.168.1.3:8080`
3. In HazardLens sidebar → **🔗 Pair / Manage CCTV Nodes** → click **📲 Show QR Code**
4. Scan the QR code with your phone's camera
5. The system automatically detects your phone's IP, verifies the camera stream, saves it to `config/paired_cameras.json`, and arms the system

### Option B — Auto-Link (Recommended)

1. Start the IP Webcam server on your phone (same Wi-Fi)
2. Click **⚡ Auto-Link** in the sidebar — HazardLens scans the entire LAN subnet, finds the first open port-8080 device, and links instantly
3. Click **🔴 ARM** — live feed starts immediately

### Changing Wi-Fi / New Location

When the demo location changes and a new Wi-Fi network is used, the phone will get a different IP address. Simply:

- Click **⚡ Auto-Link** again — the system re-discovers the new IP automatically, **no manual editing required**
- Or update `config/paired_cameras.json` manually:

```json
{
  "CAM 01": {
    "last_ip_port": "192.168.X.X:8080",
    "user": "",
    "password": "",
    "updated_at": "2026-11-16T12:50:00"
  }
}
```

> `config/paired_cameras.json` is in `.gitignore` to prevent leaking private camera IPs.  
> A safe template is provided at `config/paired_cameras.json.example`.

---

## ⚙️ Configuration

### Detection Thresholds

All constants are defined at the top of [`app.py`](app.py) for easy tuning:

| Parameter | Default | Constant | Description |
|---|---|---|---|
| Fire/Smoke Confidence | `0.80` | `FIRE_CONF_THRESHOLD` | Minimum confidence score for fire/smoke detection |
| Crowd Threshold | `1 person` | `CROWD_THRESHOLD` | Person count limit before triggering crowd alert |
| Frame Skip | `4 frames` | `FRAME_SKIP` | Process every Nth frame for real-time edge throughput |
| Email Cooldown | `120 s` | `EMAIL_COOLDOWN_SEC` | Minimum interval between email dispatches per hazard type |
| Audio Cooldown | `120 s` | `AUDIO_COOLDOWN_SEC` | Minimum interval between new voice broadcasts per hazard type |
| Alert Clear Grace | `3.0 s` | `ALERT_CLEAR_GRACE_SEC` | Debounce window to prevent alert banner flickering |
| Silence Duration | `60 s` | `SILENCE_DURATION_SEC` | Suppression window when operator presses 🔕 Silence |

### Audio Alert Files

Two distinct pre-generated Hindi voice alerts are stored in the project root:

| File | Alert Type | Voice |
|---|---|---|
| `fire_alert.mp3` | 🔥 Fire / Smoke detected | `hi-IN-SwaraNeural` |
| `crowd_alert.mp3` | 👥 Overcrowding detected | `hi-IN-SwaraNeural` |

Both are auto-generated by `edge-tts` on first startup if missing. They loop **infinitely** until the operator presses **🔕 Silence CAM 01** or clicks **⬛ DISARM**.

> Audio files are excluded from git via `.gitignore` (`*.mp3`).

### Run Tests

```bash
# Run the full test suite (61 tests)
pytest -v

# Validate Resend email configuration
python scripts/check_resend.py

# Send a live test alert email
python scripts/check_resend.py --send
```

---

## 🐳 Docker Deployment

```bash
# Build and start the container
docker compose up --build

# Access the dashboard
# http://localhost:8501
```

Ensure your `.env` file is populated before running Docker. The `docker-compose.yml` mounts it into the container.

---

## 🖥️ Usage

1. **Launch** the app: `streamlit run app.py`
2. **Authenticate** as Operator in the sidebar using your PIN (default: `1234`) to unlock ARM/DISARM/SILENCE controls
3. **Pair your camera** — click ⚡ Auto-Link or scan the QR code (see [Camera Pairing](#-camera-pairing)), or select **🎬 Tactical Demo Simulation** to test with the built-in demo video
4. **ARM** the system by clicking the 🔴 **ARM** button
5. The live feed from **CAM 01** will begin processing with dual AI engines
6. When a hazard is detected:
   - A **visual alert banner** animates on-screen (red for fire, amber for crowd)
   - A **Hindi voice alert loops continuously** (distinct audio for fire vs overcrowding)
   - An **email alert** is dispatched to the admin (if Resend is configured)
7. Click **🔕 Silence CAM 01** to mute active alerts for 60 seconds
8. Click **⬛ DISARM** to stop all monitoring and clear alerts

---

## 📂 Project Structure

```
HazardLens/
├── alerts/
│   ├── __init__.py              # Alerts package initializer
│   └── email_alerts.py          # Resend API queued background worker with retries
├── assets/
│   └── demo_feed.mp4            # Offline demo surveillance video stream
├── config/
│   ├── paired_cameras.json      # Live camera registry — IP, credentials, last seen (gitignored)
│   └── paired_cameras.json.example  # Safe template for version control
├── scripts/
│   └── check_resend.py          # Standalone Resend configuration & live email verification
├── tests/
│   ├── __init__.py              # Tests package initializer
│   ├── test_audio_alerts.py     # Unit tests for looping audio, silence/disarm state resets
│   ├── test_email_alerts.py     # Unit tests for email cooldown, retries, and missing env
│   └── test_security.py         # Automated tests for SSRF, PIN auth, QR pairing & model SHA-256
├── utils/
│   ├── __init__.py              # Utilities package initializer
│   ├── qr_pairing.py            # QR code generation, LAN camera discovery, IP pairing handler
│   └── security.py              # SSRF URL validator, PIN check, model hasher, camera registry
├── .streamlit/
│   └── config.toml              # Production hardening (CORS, XSRF, error masking)
├── app.py                       # Main application — UI, inference, alerts, camera worker
├── best.pt                      # Custom-trained fire/smoke YOLOv8 model (~6 MB)
├── fire_alert.mp3               # Pre-generated Hindi fire alert audio (gitignored)
├── crowd_alert.mp3              # Pre-generated Hindi overcrowding alert audio (gitignored)
├── Dockerfile                   # Production Docker image definition
├── docker-compose.yml           # One-command deployment configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment configuration template
├── .env                         # Local credentials (gitignored)
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

---

## 🎯 Use Cases

- 🏙️ **Smart Cities & Urban Surveillance** — Citywide hazard monitoring at intersections and public spaces
- 🎓 **University Campuses** — Gate entry monitoring and crowd management during events
- 🚉 **Railway Stations & Airports** — Platform overcrowding detection and fire safety
- 🏬 **Shopping Malls & Retail** — Real-time occupancy monitoring and emergency response
- 🏭 **Industrial Facilities** — Warehouse fire detection and worker safety compliance
- 🎪 **Public Events & Concerts** — Crowd density tracking and stampede prevention

---

## 🔮 Roadmap

- [x] QR code wireless camera pairing
- [x] Auto-IP healing (auto-discovery on new Wi-Fi)
- [x] Infinite looping Hindi voice alerts (fire + overcrowding, distinct audio)
- [x] Docker containerization for one-command deployment
- [x] Fragment-safe UI (no CSS wipe on button callbacks)
- [ ] Multi-camera simultaneous AI processing
- [ ] Cloud-based alert dashboard with historical logs
- [ ] Weapon / suspicious object detection model integration
- [ ] Mobile push notifications via Firebase / Telegram
- [ ] Recording & playback of flagged incident clips
- [ ] RTSP stream support for enterprise CCTV systems

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m "feat: add your feature"`
4. **Push** to the branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request

---

## 👨‍💻 Author

**Tushar Singh**

B.Tech CSE Student · AI/ML & Software Development Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-tusharsingh--sde-181717?style=flat-square&logo=github)](https://github.com/tusharsingh-sde)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ and a lot of ☕ by <a href="https://github.com/tusharsingh-sde">Tushar Singh</a></sub>
</p>
