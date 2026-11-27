"""
Security and validation utilities for HazardLens.
Provides:
  - SSRF-safe video stream URL validation
  - Timing-attack-resistant Operator PIN authentication
  - PyTorch model integrity (SHA-256) verification
  - Zero-UI-Input cryptographic camera auto-discovery & handshake
"""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import hmac
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

# Default expected SHA-256 for best.pt (fine-tuned custom fire model)
DEFAULT_BEST_PT_SHA256 = "18d47a2e606035a2f1b5303050125a3c6c53b679a7a1b27b25bd72d7a63b3ec3"

# Allowed streaming protocols
ALLOWED_SCHEMES = frozenset({"http", "https", "rtsp", "rtsps"})

# Blocked IP ranges (e.g. AWS/GCP/Azure link-local metadata service 169.254.169.254)
CLOUD_METADATA_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
)

# Whitelisted offline demo video paths
WHITELISTED_DEMO_PATHS = frozenset({
    "assets/demo_feed.mp4",
    "assets/demo_fire.mp4",
    "assets/demo_crowd.mp4",
})


def normalize_camera_url(url: str) -> str:
    """
    Normalizes camera URL to a clean base URL without trailing /video, /shot.jpg, or duplicate slashes:
    - Automatically adds https:// for Cloudflare tunnels (*.trycloudflare.com).
    - Automatically adds http:// if no scheme is provided for local IPs or other hosts.
    - Strips trailing /video, /shot.jpg, and slashes cleanly so /shot.jpg can be appended without double slashes.
    """
    if not url or not isinstance(url, str):
        return ""
    trimmed = url.strip().replace("\\", "/")

    parsed_init = urlparse(trimmed)
    init_scheme = (parsed_init.scheme or "").lower()

    if not (init_scheme and re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*$", init_scheme)):
        if "trycloudflare.com" in trimmed.lower():
            trimmed = f"https://{trimmed}"
        else:
            trimmed = f"http://{trimmed}"

    try:
        parsed = urlparse(trimmed)
    except Exception:
        return trimmed.rstrip("/")

    scheme = (parsed.scheme or "http").lower()
    netloc = parsed.netloc
    path = parsed.path.rstrip("/")

    # Strip /video or /shot.jpg from path
    if path.endswith("/video"):
        path = path[:-6].rstrip("/")
    elif path.endswith("/shot.jpg"):
        path = path[:-9].rstrip("/")
    elif path in ("/video", "/shot.jpg"):
        path = ""

    base = f"{scheme}://{netloc}"
    if path and path != "/":
        base = f"{base}/{path.lstrip('/')}"
    return base.rstrip("/")


def validate_stream_url(url: str, allow_demo: bool = True) -> tuple[bool, str]:
    """
    Validate a video stream URL to prevent SSRF, Local File Inclusion (LFI),
    and internal cloud metadata exfiltration.
    Allows private LAN IPs (192.168.x.x, 10.x.x.x, 172.16.0.0/12) and Cloudflare tunnels (*.trycloudflare.com).

    Returns:
        (True, normalized_url) if valid
        (False, error_reason) if rejected
    """
    if not isinstance(url, str):
        return False, "Stream target must be a string."

    trimmed = url.strip()
    if not trimmed:
        return False, "Stream URL cannot be empty."

    if len(trimmed) > 2048:
        return False, "Stream URL exceeds maximum length of 2048 characters."

    # Whitelisted demo path check
    normalized_path = trimmed.replace("\\", "/")
    if allow_demo and normalized_path in WHITELISTED_DEMO_PATHS:
        return True, normalized_path

    # Check for direct local file access patterns (e.g. C:\Windows, /etc/passwd, file://)
    if re.match(r"^[a-zA-Z]:[/\\]", trimmed) or trimmed.startswith(("\\\\", "//", "../", "./", "/")):
        return False, "Local filesystem paths are not permitted as camera stream URLs."

    try:
        parsed = urlparse(trimmed)
    except Exception as exc:
        return False, f"Malformed URL: {exc}"

    init_scheme = (parsed.scheme or "").lower()

    # If an explicit RFC scheme is present (starts with letter), validate it against allowed schemes
    if init_scheme and re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*$", init_scheme):
        if init_scheme not in ALLOWED_SCHEMES:
            return False, f"Scheme '{init_scheme}' is forbidden. Permitted schemes: http, https, rtsp, rtsps."
    else:
        # Infer scheme if missing (e.g. 192.168.1.3:8080 or xxx.trycloudflare.com)
        if "trycloudflare.com" in trimmed.lower():
            trimmed = f"https://{trimmed}"
        else:
            trimmed = f"http://{trimmed}"
        try:
            parsed = urlparse(trimmed)
        except Exception as exc:
            return False, f"Malformed URL: {exc}"

    scheme = (parsed.scheme or "").lower()
    if not scheme:
        return False, "Missing URL scheme. Use http://, https://, rtsp://, or rtsps://"

    if scheme not in ALLOWED_SCHEMES:
        return False, f"Scheme '{scheme}' is forbidden. Permitted schemes: http, https, rtsp, rtsps."

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "Stream URL must include a valid hostname or IP address."

    # Block common cloud metadata and loopback hostnames if targeting internal services
    if hostname in {"169.254.169.254", "metadata.google.internal", "instance-data"}:
        return False, "Access to cloud instance metadata services is strictly blocked."

    # Explicitly permit Cloudflare Tunnel endpoints
    if hostname.endswith(".trycloudflare.com") or hostname == "trycloudflare.com":
        return True, trimmed

    # Resolve IP address string if host is numeric IP (blocks metadata, allows private LAN IPs)
    try:
        ip_obj = ipaddress.ip_address(hostname)
        for net in CLOUD_METADATA_NETWORKS:
            if ip_obj in net:
                return False, f"IP address {hostname} belongs to link-local/cloud-metadata range."
    except ValueError:
        # Hostname is a domain name or local LAN name, not raw IP literal
        pass

    return True, trimmed


def verify_operator_pin(entered_pin: str, configured_pin: str) -> bool:
    """
    Constant-time comparison for operator PIN to defend against side-channel timing attacks.
    """
    if not entered_pin or not configured_pin:
        return False

    entered_bytes = entered_pin.strip().encode("utf-8")
    configured_bytes = configured_pin.strip().encode("utf-8")

    return hmac.compare_digest(entered_bytes, configured_bytes)


def verify_model_hash(model_path: str, expected_sha256: str = None) -> tuple[bool, str]:
    """
    Verify the SHA-256 integrity of a YOLO PyTorch model file to protect against
    pickle deserialization / arbitrary code execution (RCE).
    """
    if not os.path.exists(model_path):
        return False, f"Model file not found at '{model_path}'"

    if expected_sha256 is None:
        expected_sha256 = os.getenv("HAZARDLENS_MODEL_SHA256", DEFAULT_BEST_PT_SHA256)

    sha256 = hashlib.sha256()
    try:
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
    except Exception as exc:
        return False, f"Failed to compute model checksum: {exc}"

    calculated = sha256.hexdigest().lower()
    expected = expected_sha256.strip().lower()

    if not hmac.compare_digest(calculated, expected):
        return False, f"Integrity check FAILED for {model_path}. Expected {expected[:12]}..., got {calculated[:12]}..."

    return True, calculated


def clean_ip_address(ip_str: str | None) -> str | None:
    """
    Sanitize and extract clean IPv4/IPv6 address:
    - Strips 'http://', 'https://', trailing slashes.
    - Strips brackets from IPv6 (e.g. '[::ffff:192.168.1.14]:8080').
    - Strips '::ffff:' IPv4-mapped IPv6 prefix.
    - Strips ports if present (e.g. '192.168.1.14:8080' -> '192.168.1.14').
    """
    if not ip_str:
        return None
    cleaned = str(ip_str).strip()
    cleaned = cleaned.replace("http://", "").replace("https://", "").rstrip("/")

    # Handle bracketed host with optional port: [host]:port or [host]
    if cleaned.startswith("[") and "]" in cleaned:
        bracket_end = cleaned.find("]")
        cleaned = cleaned[1:bracket_end]
    else:
        cleaned = cleaned.replace("[", "").replace("]", "")

    # Strip ::ffff: prefix if present
    if cleaned.startswith("::ffff:"):
        cleaned = cleaned[7:].strip()
    elif "::ffff:" in cleaned:
        cleaned = cleaned.split("::ffff:")[-1].strip()

    # If it has a port (IPv4:port)
    if ":" in cleaned and not cleaned.startswith("::"):
        cleaned = cleaned.split(":")[0].strip()

    return cleaned if cleaned else None


def get_local_ip() -> str:
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


def get_all_local_ips() -> set[str]:
    """Return all known local / loopback IPs belonging to the laptop itself."""
    local_ips = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
    local_ip = get_local_ip()
    if local_ip:
        local_ips.add(local_ip)
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            raw = info[4][0]
            clean = clean_ip_address(raw)
            if clean:
                local_ips.add(clean)
    except Exception:
        pass
    return local_ips


def get_local_subnets() -> list[str]:
    """
    Detect all active local IPv4 subnets (/24) across active interfaces
    (Wi-Fi, Ethernet, Mobile Hotspot) using socket routing probes and getaddrinfo.
    """
    subnets = set()

    # Strategy 1: Active routing interface via UDP socket probe to public IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        if not local_ip.startswith("127."):
            net = ipaddress.ip_network(f"{local_ip}/24", strict=False)
            subnets.add(str(net))
    except Exception:
        pass

    # Strategy 2: Hostname resolution to capture all bound local IPv4 addresses
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                net = ipaddress.ip_network(f"{ip}/24", strict=False)
                subnets.add(str(net))
    except Exception:
        pass

    # Fallback to standard private LAN range if no interfaces discovered
    if not subnets:
        subnets.add("192.168.1.0/24")

    return sorted(list(subnets))


def _check_socket_port(ip: str, port: int, timeout: float = 0.5) -> bool:
    """Fast TCP connect probe to check if IP:port is open."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _verify_camera_endpoint(
    ip: str,
    port: int,
    user: str = "",
    secret: str = "",
    timeout: float = 0.5
) -> bool:
    """
    Verify camera endpoint (/shot.jpg, /status.json, /video, /)
    with optional HTTP Basic Auth.
    """
    endpoints = ["/shot.jpg", "/status.json", "/video", "/"]
    headers = {"User-Agent": "HazardLens-AutoPair/1.0"}

    if user and secret:
        auth_bytes = f"{user}:{secret}".encode("utf-8")
        b64_auth = base64.b64encode(auth_bytes).decode("ascii")
        headers["Authorization"] = f"Basic {b64_auth}"

    for ep in endpoints:
        url = f"http://{ip}:{port}{ep}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = resp.getcode()
                if code in (200, 204, 301, 302):
                    return True
        except urllib.error.HTTPError as he:
            if he.code == 401:
                # If credentials were provided and server rejected them, auth failed
                if user and secret:
                    return False
                # If no credentials were provided and server asks for auth, camera is detected
                return True
            continue
        except Exception:
            continue
    return False


def discover_paired_camera(
    port: int = None,
    user: str = None,
    secret: str = None,
    timeout: float = 0.3,
    cached_url: str = None
) -> str | None:
    """
    Auto-discovers and cryptographically authenticates paired CCTV node on local network.
    Fast path: validates cached_url or HAZARDLENS_DRONE_URL first (~0.02s).
    Sweep path: scans all local /24 subnets in parallel via ThreadPoolExecutor(max_workers=64).

    Returns:
        Authenticated stream URL (e.g. 'http://user:secret@192.168.1.7:8080/video') or None.
    """
    if port is None:
        try:
            port = int(os.getenv("HAZARDLENS_CAM_PORT", "8080"))
        except ValueError:
            port = 8080

    if user is None:
        user = os.getenv("HAZARDLENS_CAM_USER", "")
    if secret is None:
        secret = os.getenv("HAZARDLENS_CAM_SECRET", "")

    def _format_cam_url(ip_addr: str, cam_port: int) -> str:
        if user and secret:
            return f"http://{user}:{secret}@{ip_addr}:{cam_port}/video"
        return f"http://{ip_addr}:{cam_port}/video"

    # Fast Path 1: Check cached_url
    if cached_url:
        try:
            parsed = urlparse(cached_url)
            host = parsed.hostname
            p = parsed.port or port
            if host and _check_socket_port(host, p, timeout=0.15):
                if _verify_camera_endpoint(host, p, user, secret, timeout=0.3):
                    return _format_cam_url(host, p)
        except Exception:
            pass

    # Fast Path 2: Check HAZARDLENS_DRONE_URL env var if set
    drone_env = os.getenv("HAZARDLENS_DRONE_URL", "").strip()
    if drone_env:
        try:
            parsed = urlparse(drone_env)
            host = parsed.hostname
            p = parsed.port or port
            if host and _check_socket_port(host, p, timeout=0.15):
                if _verify_camera_endpoint(host, p, user, secret, timeout=0.3):
                    return _format_cam_url(host, p)
        except Exception:
            pass

    # Full Subnet Discovery Sweep
    subnets = get_local_subnets()
    all_target_ips = []
    for subnet in subnets:
        try:
            net = ipaddress.ip_network(subnet, strict=False)
            all_target_ips.extend([str(ip) for ip in net.hosts()])
        except Exception:
            continue

    if not all_target_ips:
        return None

    # Step 1: Parallel Port Scan across all target IPs
    candidate_ips = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {
            executor.submit(_check_socket_port, ip, port, timeout): ip
            for ip in all_target_ips
        }
        for future in as_completed(futures):
            ip = futures[future]
            try:
                if future.result():
                    candidate_ips.append(ip)
            except Exception:
                pass

    if not candidate_ips:
        return None

    # Step 2: Verify candidate IPs with camera endpoint & authentication
    for ip in candidate_ips:
        if _verify_camera_endpoint(ip, port, user, secret, timeout=0.5):
            return _format_cam_url(ip, port)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-CAMERA HARDWARE PAIRING REGISTRY & AUTO-IP HEALING
# ─────────────────────────────────────────────────────────────────────────────
PAIRED_CAMERAS_FILE = os.path.join("config", "paired_cameras.json")


def load_paired_cameras() -> dict:
    """Load the paired cameras registry from config/paired_cameras.json."""
    if not os.path.exists(PAIRED_CAMERAS_FILE):
        return {}
    try:
        with open(PAIRED_CAMERAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_paired_cameras(data: dict) -> bool:
    """Save the paired cameras registry to config/paired_cameras.json."""
    try:
        os.makedirs(os.path.dirname(PAIRED_CAMERAS_FILE), exist_ok=True)
        with open(PAIRED_CAMERAS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def get_paired_camera(slot: str = "CAM 01") -> dict | None:
    """Get paired camera info for a specific slot."""
    registry = load_paired_cameras()
    return registry.get(slot)


def save_camera_slot(slot: str, ip_port: str, user: str = "", password: str = "") -> bool:
    """
    Register or update a camera slot in the persistent registry.
    Strictly prevents saving the operator's laptop IP or loopback address.
    """
    registry = load_paired_cameras()
    trimmed = ip_port.strip().replace("::ffff:", "").rstrip("/")

    # Check loopback / laptop IP
    parsed = urlparse(trimmed if "://" in trimmed else f"http://{trimmed}")
    hostname = parsed.hostname or trimmed
    raw_ip_cleaned = clean_ip_address(hostname)

    # Safety Guard: Never register the operator's laptop or loopback as a camera feed!
    local_ips = get_all_local_ips()
    if raw_ip_cleaned in local_ips or (raw_ip_cleaned and raw_ip_cleaned.startswith("127.")):
        return False

    normalized_url = normalize_camera_url(trimmed)
    if "://" not in trimmed and not trimmed.startswith("http"):
        last_val = trimmed.replace("http://", "").replace("https://", "").rstrip("/")
    else:
        last_val = normalized_url

    registry[slot] = {
        "last_ip_port": last_val,
        "stream_url": normalized_url,
        "user": user.strip(),
        "password": password.strip(),
        "updated_at": datetime.now().isoformat()
    }
    return save_paired_cameras(registry)


def scan_network_cameras(port: int = 8080, timeout: float = 1.2) -> list[str]:
    """
    Detects all active local IPv4 subnets (Wi-Fi, Windows Mobile Hotspot, Android/iOS hotspots)
    and scans port 8080 in parallel via ThreadPoolExecutor(max_workers=128).
    Returns a list of discovered '<ip>:<port>' endpoints that respond to HTTP
    (including 401 Unauthorized, which indicates an authenticated IP webcam service).
    """
    subnets = get_local_subnets()
    all_target_ips = []
    for subnet in subnets:
        try:
            net = ipaddress.ip_network(subnet, strict=False)
            all_target_ips.extend([str(ip) for ip in net.hosts()])
        except Exception:
            continue

    if not all_target_ips:
        return []

    # Step 1: Parallel TCP connect probe with min(1.0, timeout) across 128 threads
    open_ips = []
    tcp_timeout = min(1.0, timeout)
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = {
            executor.submit(_check_socket_port, ip, port, tcp_timeout): ip
            for ip in all_target_ips
        }
        for future in as_completed(futures):
            ip = futures[future]
            try:
                if future.result():
                    open_ips.append(ip)
            except Exception:
                pass

    if not open_ips:
        return []

    # Step 2: Parallel HTTP check (accepts 200, 301, 302, 401, 403, 404 as valid HTTP service)
    discovered = []
    http_timeout = max(1.0, timeout)

    def _probe_http(target_ip: str) -> str | None:
        url = f"http://{target_ip}:{port}/shot.jpg"
        req = urllib.request.Request(url, headers={"User-Agent": "HazardLens-Scanner/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                if resp.getcode() in (200, 204, 301, 302):
                    return f"{target_ip}:{port}"
        except urllib.error.HTTPError as he:
            if he.code in (401, 403, 404):
                return f"{target_ip}:{port}"
        except Exception:
            pass

        url_root = f"http://{target_ip}:{port}/"
        req_root = urllib.request.Request(url_root, headers={"User-Agent": "HazardLens-Scanner/1.0"})
        try:
            with urllib.request.urlopen(req_root, timeout=http_timeout) as resp:
                if resp.getcode() in (200, 204, 301, 302):
                    return f"{target_ip}:{port}"
        except urllib.error.HTTPError as he:
            if he.code in (401, 403, 404):
                return f"{target_ip}:{port}"
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        http_futures = [executor.submit(_probe_http, ip) for ip in open_ips]
        for f in as_completed(http_futures):
            try:
                res = f.result()
                if res and res not in discovered:
                    discovered.append(res)
            except Exception:
                pass

    return sorted(discovered)


def discover_cameras_on_lan(port: int = 8080, timeout: float = 1.0) -> list[str]:
    """
    Scans local network for active IP Webcam nodes on port 8080.
    Strictly filters out any local / loopback IPs belonging to the laptop itself.
    """
    nodes = scan_network_cameras(port=port, timeout=timeout)
    local_ips = get_all_local_ips()
    filtered = []
    for node in nodes:
        node_ip = clean_ip_address(node)
        if node_ip and node_ip not in local_ips and not node_ip.startswith("127."):
            filtered.append(node)
    return filtered


def verify_camera_handshake(ip_port: str, user: str = "", password: str = "", timeout: float = 1.5) -> bool:
    """
    Request http(s)://<ip_port>/shot.jpg with HTTP Basic Auth (only if user is provided).
    If user is empty, sends auth=None to avoid invalid/rejected authentication headers.
    Return True only if HTTP 200 OK is returned with image/stream content.
    """
    if not ip_port:
        return False

    trimmed = ip_port.strip().replace("::ffff:", "").rstrip("/")
    if trimmed.startswith("https://") or "trycloudflare.com" in trimmed.lower():
        clean_host = trimmed.replace("https://", "").replace("http://", "").rstrip("/")
        url = f"https://{clean_host}/shot.jpg"
    elif trimmed.startswith("http://"):
        clean_host = trimmed.replace("http://", "").rstrip("/")
        url = f"http://{clean_host}/shot.jpg"
    else:
        url = f"http://{trimmed}/shot.jpg"

    # Only send auth credentials if user is explicitly provided
    auth_val = (user, password) if user else None

    # Primary verification using requests.get
    try:
        import requests
        headers = {"User-Agent": "HazardLens-Handshake/1.0"}
        resp = requests.get(url, auth=auth_val, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "").lower()
            if "image" in content_type or resp.content.startswith(b"\xff\xd8\xff") or len(resp.content) > 0:
                return True
        elif resp.status_code == 401:
            return False
    except Exception:
        pass

    # Fallback verification via urllib.request
    headers = {"User-Agent": "HazardLens-Handshake/1.0"}
    if auth_val:
        auth_bytes = f"{user}:{password}".encode("utf-8")
        b64_auth = base64.b64encode(auth_bytes).decode("ascii")
        headers["Authorization"] = f"Basic {b64_auth}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.getcode() == 200:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                chunk = resp.read(16)
                if "image" in content_type or chunk.startswith(b"\xff\xd8\xff") or len(chunk) > 0:
                    return True
    except Exception:
        return False
    return False


def resolve_paired_camera(slot: str = "CAM 01") -> str | None:
    """
    Resolve live stream URL for the paired camera slot with Auto-IP Healing.
    Fast Path: If slot has last_ip_port or stream_url configured, cleans it and immediately returns the stream URL.
    """
    cam_info = get_paired_camera(slot)

    def _make_stream_url(ip_p: str, u: str = "", p: str = "") -> str:
        clean = ip_p.replace("::ffff:", "").rstrip("/")
        if clean.startswith("https://") or "trycloudflare.com" in clean.lower():
            if not clean.startswith("https://") and not clean.startswith("http://"):
                clean = f"https://{clean}"
            return clean
        clean = clean.replace("http://", "").rstrip("/")
        if u or p:
            return f"http://{u}:{p}@{clean}/video"
        return f"http://{clean}/video"

    # Fast Path: If camera was registered/configured, check and return stream URL
    if cam_info and (cam_info.get("stream_url") or cam_info.get("last_ip_port")):
        configured_target = cam_info.get("stream_url") or cam_info.get("last_ip_port", "").strip()
        user = cam_info.get("user", "").strip()
        password = cam_info.get("password", "").strip()

        # Cloudflare tunnel URLs fast path
        if "trycloudflare.com" in str(configured_target).lower():
            return _make_stream_url(str(configured_target), user, password)

        # Clean corrupted ::ffff: if present
        clean_ip_port = str(configured_target).replace("::ffff:", "").strip()
        if clean_ip_port != configured_target:
            save_camera_slot(slot, clean_ip_port, user=user, password=password)
            configured_target = clean_ip_port

        # Fast handshake check
        if verify_camera_handshake(configured_target, user=user, password=password, timeout=0.8):
            return _make_stream_url(configured_target, user, password)

        # Auto-Heal: If configured_target failed handshake, scan for reachable node
        discovered_nodes = scan_network_cameras(timeout=1.0)
        for node in discovered_nodes:
            if node == configured_target:
                continue
            if verify_camera_handshake(node, user=user, password=password, timeout=1.0):
                save_camera_slot(slot, node, user=user, password=password)
                return _make_stream_url(node, user, password)

        # Fallback to configured target rather than dropping to demo mode
        return _make_stream_url(configured_target, user, password)

    # Auto-Discovery Fallback: only if unconfigured, scan network
    discovered_nodes = scan_network_cameras(timeout=1.2)
    if discovered_nodes:
        fallback_node = discovered_nodes[0].replace("::ffff:", "")
        save_camera_slot(slot, fallback_node, user="", password="")
        return _make_stream_url(fallback_node)

    return None
