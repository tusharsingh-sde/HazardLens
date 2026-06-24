# 🚨 HazardLens

### Autonomous Hazard Detection & Emergency Response Platform

HazardLens is an AI-powered surveillance system that transforms traditional CCTV cameras into intelligent hazard monitoring agents capable of detecting emergencies in real time.

The system continuously monitors live video feeds, detects hazards such as fire, smoke, and overcrowding, and instantly generates visual and voice alerts to reduce emergency response time.

---

## 📌 Problem Statement

Traditional CCTV systems are passive monitoring tools.

In most emergency situations, incidents are detected only after a human operator notices them, verifies them, and manually escalates the situation. This delay can lead to increased damage, risk to human life, and slower emergency response.

HazardLens aims to bridge this gap by providing real-time automated hazard detection and alert generation.

---

## ✨ Features

- 🔥 Real-time Fire Detection
- 🌫️ Smoke Detection
- 👥 Crowd Monitoring & Overcrowding Alerts
- 🎙️ Hindi Voice Announcements using Sarvam AI
- 📹 IP Camera / CCTV Feed Support
- 🖥️ Tactical Command Center Dashboard
- 🚨 Automated Hazard Alert Generation
- ⚡ Real-time AI Inference Pipeline

---

## 🏗️ System Architecture

```text
Live CCTV Feed
        │
        ▼
  Video Processing
        │
 ┌──────┴──────┐
 ▼             ▼

Fire Model   Crowd Model
 (YOLO)       (YOLOv8)

 ▼             ▼

Hazard Detection Engine
          │
          ▼

     Alert Manager
          │
    ┌─────┴─────┐
    ▼           ▼

Visual Alert  Voice Alert
              (Sarvam AI)
```

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Computer Vision
- OpenCV
- YOLOv8
- Custom Fire Detection Model

### AI Services
- Sarvam AI Text-to-Speech

### Backend
- Python

### Utilities
- NumPy
- Requests
- Dotenv

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/HazardLens.git

cd HazardLens
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
SARVAM_API_KEY=YOUR_API_KEY
```

### Run Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Fire Detection

![Fire Detection](screenshots/fire-alert.png)

### Crowd Detection

![Crowd Detection](screenshots/crowd-alert.png)

---

## 🎯 Potential Applications

- Smart Cities
- University Campuses
- Railway Stations
- Airports
- Shopping Malls
- Industrial Facilities
- Public Event Monitoring
- Traffic Surveillance

---

## 🔮 Future Roadmap

- Automated Fire Brigade Dispatch
- Police Dispatch Integration
- Emergency SMS/WhatsApp Alerts
- GIS-Based Incident Mapping
- Multi-Camera Incident Correlation
- Severity Scoring Engine
- Incident Reporting Dashboard

---

## 💡 Engineering Challenges Solved

- Real-time CCTV stream processing
- Dual-model AI inference architecture
- Crowd density estimation
- Hazard event classification
- Automated voice alert generation
- Low-latency monitoring dashboard

---

## 👨‍💻 Author

**Tushar Singh**

B.Tech CSE Student

Passionate about AI, Computer Vision, Smart Infrastructure and Public Safety Systems.

GitHub: https://github.com/tusharsingh-sde

---

## 📜 License

This project is licensed under the MIT License.
