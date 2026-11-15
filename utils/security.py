"""
Security and validation utilities for HazardLens.
Provides:
  - SSRF-safe video stream URL validation
  - Timing-attack-resistant Operator PIN authentication
  - PyTorch model integrity (SHA-256) verification
"""
import hashlib
import hmac
import ipaddress
import os
import re
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


def validate_stream_url(url: str, allow_demo: bool = True) -> tuple[bool, str]:
    """
    Validate a video stream URL to prevent SSRF, Local File Inclusion (LFI),
    and internal cloud metadata exfiltration.

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

    # Resolve IP address string if host is numeric IP
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
