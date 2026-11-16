"""
QR Code Camera Pairing Server for HazardLens.
Enables instant WhatsApp-Web style zero-UI-input camera pairing over Wi-Fi.
Primary pairing workflow runs directly on the main Streamlit port (8501)
via st.query_params to avoid firewall issues, with background port 8502 as fallback.
"""
import http.server
from io import BytesIO
import json
import os
import qrcode
import secrets
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

from utils.security import save_camera_slot, get_paired_camera

ACTIVE_PAIRING_TOKENS = {}
_PAIRING_LOCK = threading.Lock()
_PAIRING_SERVER = None
_PAIRING_PORT = 8502


def get_laptop_wifi_ip() -> str:
    """Detect the active Wi-Fi / LAN IP of the laptop."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


class PairingHTTPHandler(http.server.BaseHTTPRequestHandler):
    """
    Handles fallback HTTP requests on port 8502 from phone scanning the QR code.
    Explicitly uses HTTP/1.1 with Content-Length and Connection: close to prevent
    ERR_TOO_MANY_RETRIES on mobile Chrome.
    """
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/pair", "/"):
            qs = parse_qs(parsed.query)
            token = qs.get("token", [""])[0] or qs.get("pair_token", [""])[0]
            slot = qs.get("slot", ["CAM 01"])[0]
            if slot == "CAM01" or slot == "CAM1":
                slot = "CAM 01"
            elif slot.startswith("CAM") and len(slot) == 5:
                slot = f"{slot[:3]} {slot[3:]}"

            phone_ip = self.client_address[0]

            with _PAIRING_LOCK:
                session = ACTIVE_PAIRING_TOKENS.get(token)

            if not session:
                err_body = b"<h3>HazardLens Security: Invalid or Expired Pairing Token.</h3>"
                self.send_response(403)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(err_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(err_body)
                return

            target_slot = session.get("slot", slot) if session else slot
            ip_port = f"{phone_ip}:8080"

            # Register camera in config/paired_cameras.json
            save_camera_slot(target_slot, ip_port, user="", password="")

            with _PAIRING_LOCK:
                if session:
                    session["paired"] = True
                    session["phone_ip"] = phone_ip
                else:
                    ACTIVE_PAIRING_TOKENS[token] = {
                        "slot": target_slot,
                        "created_at": time.time(),
                        "paired": True,
                        "phone_ip": phone_ip,
                    }

            # Mobile response landing page
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HazardLens · Node Paired</title>
    <style>
        body {{
            background: #0B0F14; color: #E6EDF3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex; align-items: center; justify-content: center;
            height: 100vh; margin: 0; padding: 20px; box-sizing: border-box; text-align: center;
        }}
        .card {{
            background: #111820; border: 1px solid #1F2B38; border-radius: 8px;
            padding: 32px 24px; max-width: 360px; width: 100%; box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        .badge {{
            display: inline-block; background: rgba(61,214,140,0.1); color: #3DD68C;
            border: 1px solid #3DD68C; border-radius: 4px; padding: 6px 14px;
            font-size: 13px; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 16px;
        }}
        h2 {{ margin: 0 0 10px; font-size: 22px; color: #3CC8E8; }}
        p {{ color: #A3B3C2; font-size: 14px; line-height: 1.5; margin: 8px 0; }}
        .ip {{ font-family: monospace; color: #3DD68C; font-size: 16px; margin: 12px 0; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="badge">&#9679; NODE CONNECTED</div>
        <h2>HazardLens Command</h2>
        <p>Your smartphone has been linked as live surveillance feed:</p>
        <div class="ip">{target_slot} &middot; {ip_port}</div>
        <p style="font-size: 12px; color: #7B8DA0; margin-top: 18px;">
            Ensure <strong>IP Webcam</strong> is broadcasting on port 8080.<br>
            You can now close this tab.
        </p>
    </div>
</body>
</html>"""
            html_bytes = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(html_bytes)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

    def log_message(self, format, *args):
        # Silence access logging to keep terminal output clean
        pass


def ensure_pairing_server_running(port: int = 8502) -> int:
    """Ensure the optional background pairing HTTP server is running on fallback port."""
    global _PAIRING_SERVER, _PAIRING_PORT
    with _PAIRING_LOCK:
        if _PAIRING_SERVER is not None:
            return _PAIRING_PORT

        for p in (port, 8503, 8504, 8505, 0):
            try:
                server = http.server.ThreadingHTTPServer(("0.0.0.0", p), PairingHTTPHandler)
                _PAIRING_PORT = server.server_port
                _PAIRING_SERVER = server
                t = threading.Thread(target=_PAIRING_SERVER.serve_forever, daemon=True)
                t.start()
                return _PAIRING_PORT
            except Exception:
                continue

        return port


def generate_qr_code_png(url: str) -> bytes:
    """Generate crisp PNG image bytes for the given URL using qrcode."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#000000", back_color="#FFFFFF")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_pairing_session(slot: str = "CAM 01", port: int = 8501) -> tuple[str, str, bytes]:
    """
    Generates a unique one-time pairing session.
    By default points to the primary Streamlit server port (8501) so Windows Firewall
    and cloud reverse proxies do not block phone connection requests.
    Returns:
        (token, pairing_url, qr_png_bytes)
    """
    # Proactively start background 8502 fallback listener
    try:
        ensure_pairing_server_running(8502)
    except Exception:
        pass

    # Detect Streamlit port from environment if available
    try:
        st_port = int(os.getenv("PORT", os.getenv("STREAMLIT_SERVER_PORT", str(port))))
    except ValueError:
        st_port = port

    laptop_ip = get_laptop_wifi_ip()
    token = secrets.token_urlsafe(16)

    with _PAIRING_LOCK:
        ACTIVE_PAIRING_TOKENS[token] = {
            "slot": slot,
            "created_at": time.time(),
            "paired": False,
            "phone_ip": None,
        }

    # Format slot query param without spaces for clean mobile URLs (e.g., CAM01)
    slot_param = slot.replace(" ", "")
    pairing_url = f"http://{laptop_ip}:{st_port}/?pair_token={token}&slot={slot_param}"
    qr_png = generate_qr_code_png(pairing_url)
    return token, pairing_url, qr_png


def get_pairing_status(token: str) -> dict | None:
    """Check status of a pairing token."""
    with _PAIRING_LOCK:
        sess = ACTIVE_PAIRING_TOKENS.get(token)
        if sess:
            return dict(sess)
        return None


def validate_pairing_token(token: str, max_age_seconds: float = 3600.0) -> bool:
    """Check if token is valid and not expired."""
    if not token or len(token) < 8:
        return False
    with _PAIRING_LOCK:
        sess = ACTIVE_PAIRING_TOKENS.get(token)
        if not sess:
            return False
        if time.time() - sess.get("created_at", 0) > max_age_seconds:
            return False
        return True


def mark_pairing_complete(token: str, phone_ip: str, slot: str = "CAM 01") -> bool:
    """Mark a pairing token as completed and record the connected phone IP."""
    with _PAIRING_LOCK:
        if token not in ACTIVE_PAIRING_TOKENS:
            ACTIVE_PAIRING_TOKENS[token] = {
                "slot": slot,
                "created_at": time.time(),
                "paired": True,
                "phone_ip": phone_ip,
            }
        else:
            ACTIVE_PAIRING_TOKENS[token]["paired"] = True
            ACTIVE_PAIRING_TOKENS[token]["phone_ip"] = phone_ip
            if slot:
                ACTIVE_PAIRING_TOKENS[token]["slot"] = slot
        return True
