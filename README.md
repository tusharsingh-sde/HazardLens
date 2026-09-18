<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-27338E?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Edge--TTS-0078D4?style=for-the-badge&logo=microsoft&logoColor=white" alt="Edge-TTS" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<h1 align="center">⬡ HazardLens</h1>

<p align="center">
  <strong>AI-Powered Smart City Surveillance Command Center</strong><br/>
  <em>Real-Time Fire Detection · Crowd Monitoring · Neural Voice Alerts · Email Dispatch</em>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#%EF%B8%8F-system-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-license">License</a>
</p>

---

## 📖 Overview

**HazardLens** is an edge-optimized, autonomous AI surveillance system that transforms traditional CCTV cameras into intelligent hazard monitoring agents. It runs a **dual YOLOv8 inference pipeline** — a custom-trained fire/smoke model alongside a general-purpose crowd counter — to continuously analyse live video feeds and trigger multi-channel emergency alerts with **zero cloud dependency**.

When a hazard is detected, the system simultaneously:

- Renders an animated tactical alert banner on the command center dashboard
- Broadcasts a **Hindi neural voice alert** via Microsoft Edge-TTS (`hi-IN-SwaraNeural`)
- Dispatches an **automated HTML email** to the configured security administrator
- Applies a **"Latest Hazard Wins"** state machine to prevent alert stacking, UI freezes, and audio collisions

---

## ❓ Problem Statement

Traditional CCTV systems are **passive monitoring tools**. In emergency situations, incidents are detected only after a human operator notices them, verifies them, and manually escalates the alarm. This critical delay increases physical damage, risks human life, and slows down emergency response.

**HazardLens** bridges this gap by providing **low-latency automated hazard detection, smart alert prioritization, and instant localized voice broadcasts** — all running at the edge.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔥 **Fire & Smoke Detection** | Custom fine-tuned YOLOv8 model with multilingual class mapping (EN/ZH) and `conf ≥ 0.68` threshold |
| 👥 **Crowd Monitoring** | Real-time person counting via YOLOv8n with configurable overcrowding threshold |
| 🎙️ **Hindi Neural Voice Alerts** | Locally powered by Microsoft Edge-TTS (`hi-IN-SwaraNeural`) — free, fast, and offline-capable |
| 📧 **Automated Email Dispatch** | Threaded HTML email alerts sent to the security admin via SMTP (non-blocking) |
| ⚡ **"Latest Hazard Wins" Engine** | Prevents UI freezes, alert looping, and audio overlapping via a finite state machine |
| 📹 **IP Camera / CCTV Support** | Direct OpenCV frame stream ingestion from any RTSP / HTTP video source |
| 🖥️ **Tactical Command Center UI** | Dark-themed, military-inspired dashboard with 4-camera grid, live metrics, and animated alert banners |
| 🎯 **False-Positive Immunity** | Optimized detection confidence thresholds to ignore shadows, lighting changes, and skin tones |

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────┐
                    │  Live CCTV / IP Camera    │
                    │  (HTTP / RTSP Stream)     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  OpenCV Frame Capture     │
                    │  (4-frame skip pipeline)  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌─────────────────┐      ┌─────────────────┐
          │  🔥 Engine 1     │      │  👥 Engine 2     │
          │  Fire / Smoke    │      │  Crowd Counter   │
          │  (best.pt ~6MB)  │      │  (YOLOv8n ~6MB)  │
          │  conf ≥ 0.68     │      │  class: person   │
          └────────┬────────┘      └────────┬────────┘
                   │                         │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │  Hazard Detection Engine  │
                   │  "Latest Hazard Wins" FSM │
                   └────────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
    │ 🖥️ Visual     │  │ 🎙️ Voice     │  │ 📧 Email Alert   │
    │ Alert Banner  │  │ Alert (TTS)  │  │ (SMTP Thread)    │
    │ (Streamlit)   │  │ (Edge-TTS)   │  │ (Background)     │
    └──────────────┘  └──────────────┘  └──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend & Dashboard** | [Streamlit](https://streamlit.io/) with custom CSS command-center theme |
| **Computer Vision** | [OpenCV](https://opencv.org/) (`opencv-python-headless`) |
| **Object Detection** | [Ultralytics YOLOv8](https://docs.ultralytics.com/) — dual model inference |
| **Fire/Smoke Model** | Custom fine-tuned YOLOv8 (`best.pt` — ~6 MB) |
| **Crowd Model** | YOLOv8n pre-trained (`yolov8n.pt` — ~6 MB, auto-downloaded) |
| **Voice Engine** | [Microsoft Edge-TTS](https://pypi.org/project/edge-tts/) (`hi-IN-SwaraNeural`) |
| **Email Alerts** | Python `smtplib` + `email.mime` (threaded, non-blocking) |
| **Runtime** | Python 3.10+ with `asyncio` |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10 or higher** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)
- **pip** (comes bundled with Python 3.10+)
- An **IP camera** or a phone running an IP webcam app (e.g., [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) on Android)

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

This will install:

| Package | Purpose |
|---|---|
| `streamlit >=1.30.0` | Dashboard framework |
| `ultralytics >=8.1.0` | YOLOv8 inference engine |
| `opencv-python-headless >=4.8.0` | Video frame processing |
| `edge-tts >=6.1.10` | Hindi neural voice synthesis |
| `numpy >=1.24.0` | Numerical array operations |
| `pillow >=10.0.0` | Image utilities |

---

### 4 · Run the Application

```bash
streamlit run app.py
```

The command center will launch in your default browser at **`http://localhost:8501`**.

---

## ⚙️ Configuration

### IP Camera Setup

1. Install an IP camera app on your phone (e.g., **IP Webcam** for Android)
2. Start the camera server on your phone — it will display a stream URL
3. In the HazardLens sidebar **Control Panel**, paste your camera's stream URL into the **Feed Source** field
   - Default: `http://192.168.1.7:8080/video`
   - Make sure your phone and computer are on the **same Wi-Fi network**

### Email Alerts (Optional)

To enable automated email dispatch on hazard detection, edit the following variables in [`app.py`](app.py) (lines 370–372):

```python
SENDER_EMAIL = "your-email@gmail.com"
APP_PASSWORD  = "your-gmail-app-password"
ADMIN_EMAIL   = "admin-recipient@example.com"
```

> **Tip:** Use a [Gmail App Password](https://support.google.com/accounts/answer/185833) (not your regular password). Enable 2-Step Verification on your Google account first, then generate an app-specific password.

### Detection Thresholds

| Parameter | Default | Location |
|---|---|---|
| Fire/Smoke Confidence | `0.68` | `app.py` line 666 |
| Crowd Threshold | `5 persons` | `app.py` line 706 |
| Frame Skip | `4 frames` | `app.py` line 655 |

---

## 🖥️ Usage

1. **Launch** the app with `streamlit run app.py`
2. **Configure** your IP camera URL in the sidebar
3. **ARM** the system by clicking the 🔴 **ARM** button
4. The live feed from **CAM 01** will begin processing with dual AI engines
5. When a hazard is detected:
   - A **visual alert banner** animates on-screen (red for fire, amber for crowd)
   - A **Hindi voice alert** plays automatically
   - An **email alert** is dispatched to the admin (if configured)
6. Click **🔕 SILENCE** to dismiss active alerts
7. Click **⬛ DISARM** to stop monitoring

---

## 📂 Project Structure

```
HazardLens/
├── app.py                 # Main application — UI, inference, alerts
├── best.pt                # Custom-trained fire/smoke YOLOv8 model
├── yolov8n.pt             # Pre-trained YOLOv8n (auto-downloaded if missing)
├── requirements.txt       # Python dependencies
├── fire_alert.mp3         # Generated Hindi TTS alert for fire events
├── temp_alert.mp3         # Generated Hindi TTS alert for crowd events
├── .gitignore             # Git ignore rules
└── README.md              # This file
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

- [ ] Multi-camera simultaneous AI processing
- [ ] Cloud-based alert dashboard with historical logs
- [ ] Weapon / suspicious object detection model integration
- [ ] Mobile push notifications via Firebase/Telegram
- [ ] Recording & playback of flagged incident clips
- [ ] Docker containerization for one-command deployment
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
