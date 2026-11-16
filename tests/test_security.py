"""
Security test suite for HazardLens:
  - SSRF & Local File Inclusion prevention
  - Operator PIN gate authentication
  - Model integrity (SHA-256) verification
"""
import os
import pytest
from utils.security import (
    validate_stream_url,
    verify_operator_pin,
    verify_model_hash,
    get_local_subnets,
    discover_paired_camera,
    scan_network_cameras,
    verify_camera_handshake,
    resolve_paired_camera,
    load_paired_cameras,
    save_paired_cameras,
    get_paired_camera,
    save_camera_slot,
    DEFAULT_BEST_PT_SHA256,
)


class TestStreamUrlValidation:
    """Tests for SSRF and Local File Inclusion defense."""

    @pytest.mark.parametrize("valid_url", [
        "http://192.168.1.3:8080/video",
        "https://cam.example.edu/stream/live.mjpg",
        "rtsp://admin:pass@192.168.1.100:554/h264",
        "rtsps://secure.campus.miet.ac.in:322/live",
    ])
    def test_valid_stream_urls_accepted(self, valid_url):
        ok, reason = validate_stream_url(valid_url)
        assert ok is True
        assert reason == valid_url

    @pytest.mark.parametrize("malicious_url", [
        "file:///etc/passwd",
        "file:///C:/Windows/win.ini",
        "javascript:alert(document.cookie)",
        "gopher://127.0.0.1:6379/_flushall",
        "ftp://anonymous@ftp.example.com/dump.tar",
        "C:\\Windows\\System32\\cmd.exe",
        "/etc/shadow",
        "../../secrets.env",
        "\\\\192.168.1.1\\share\\payload.exe",
    ])
    def test_local_file_inclusion_and_bad_schemes_rejected(self, malicious_url):
        ok, reason = validate_stream_url(malicious_url)
        assert ok is False
        assert len(reason) > 0

    @pytest.mark.parametrize("metadata_url", [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.1.1/internal",
        "https://metadata.google.internal/computeMetadata/v1/",
        "http://instance-data/latest/meta-data/",
    ])
    def test_cloud_metadata_endpoints_blocked(self, metadata_url):
        ok, reason = validate_stream_url(metadata_url)
        assert ok is False
        assert "metadata" in reason.lower() or "blocked" in reason.lower() or "link-local" in reason.lower()

    def test_whitelisted_demo_path_allowed(self):
        ok, res = validate_stream_url("assets/demo_feed.mp4", allow_demo=True)
        assert ok is True
        assert res == "assets/demo_feed.mp4"

    def test_empty_and_oversized_urls_rejected(self):
        ok_empty, _ = validate_stream_url("")
        assert ok_empty is False

        ok_whitespace, _ = validate_stream_url("   ")
        assert ok_whitespace is False

        oversized = "http://example.com/" + "a" * 2100
        ok_long, reason = validate_stream_url(oversized)
        assert ok_long is False
        assert "exceeds maximum length" in reason


class TestOperatorPinVerification:
    """Tests for Operator Authentication gate and timing attack resistance."""

    def test_correct_pin_succeeds(self):
        assert verify_operator_pin("1234", "1234") is True
        assert verify_operator_pin("secret-pin-99", "secret-pin-99") is True

    def test_incorrect_pin_fails(self):
        assert verify_operator_pin("wrong", "1234") is False
        assert verify_operator_pin("1235", "1234") is False

    def test_empty_or_none_pin_fails(self):
        assert verify_operator_pin("", "1234") is False
        assert verify_operator_pin("1234", "") is False
        assert verify_operator_pin(None, "1234") is False

    def test_whitespace_trimmed_properly(self):
        assert verify_operator_pin("  1234  ", "1234") is True


class TestModelIntegrityVerification:
    """Tests for PyTorch model SHA-256 integrity verification."""

    def test_best_pt_hash_matches_known_checksum(self):
        if not os.path.exists("best.pt"):
            pytest.skip("best.pt not found in workspace")
        ok, val = verify_model_hash("best.pt", DEFAULT_BEST_PT_SHA256)
        assert ok is True
        assert val == DEFAULT_BEST_PT_SHA256

    def test_tampered_hash_fails(self):
        if not os.path.exists("best.pt"):
            pytest.skip("best.pt not found in workspace")
        bogus_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        ok, msg = verify_model_hash("best.pt", bogus_hash)
        assert ok is False
        assert "FAILED" in msg

    def test_nonexistent_model_file_fails(self):
        ok, msg = verify_model_hash("nonexistent_model.pt", DEFAULT_BEST_PT_SHA256)
        assert ok is False
        assert "not found" in msg


class TestCameraAutoPairing:
    """Tests for zero-UI-input camera auto-discovery and cryptographic handshake."""

    def test_get_local_subnets_returns_valid_cidrs(self):
        subnets = get_local_subnets()
        assert isinstance(subnets, list)
        assert len(subnets) > 0
        import ipaddress
        for net_str in subnets:
            net = ipaddress.ip_network(net_str, strict=False)
            assert net.version == 4

    def test_discover_paired_camera_fast_path_cached(self, monkeypatch):
        monkeypatch.setattr("utils.security._check_socket_port", lambda host, port, timeout: True)
        monkeypatch.setattr("utils.security._verify_camera_endpoint", lambda host, port, user, secret, timeout: True)

        url = discover_paired_camera(
            port=8080,
            user="cam01",
            secret="hz_secret_9921",
            cached_url="http://192.168.1.3:8080/video"
        )
        assert url == "http://cam01:hz_secret_9921@192.168.1.3:8080/video"

    def test_discover_paired_camera_fast_path_drone_env(self, monkeypatch):
        monkeypatch.setenv("HAZARDLENS_DRONE_URL", "http://192.168.1.55:8080/video")
        monkeypatch.setattr("utils.security._check_socket_port", lambda host, port, timeout: True)
        monkeypatch.setattr("utils.security._verify_camera_endpoint", lambda host, port, user, secret, timeout: True)

        url = discover_paired_camera(port=8080, user="drone", secret="pass")
        assert url == "http://drone:pass@192.168.1.55:8080/video"

    def test_discover_paired_camera_subnet_sweep(self, monkeypatch):
        monkeypatch.delenv("HAZARDLENS_DRONE_URL", raising=False)
        monkeypatch.setattr("utils.security.get_local_subnets", lambda: ["192.168.10.0/24"])

        def mock_check(ip, port, timeout):
            return ip == "192.168.10.42"

        def mock_verify(ip, port, user, secret, timeout):
            return ip == "192.168.10.42"

        monkeypatch.setattr("utils.security._check_socket_port", mock_check)
        monkeypatch.setattr("utils.security._verify_camera_endpoint", mock_verify)

        url = discover_paired_camera(port=8080, user="cam01", secret="hz_secret_9921")
        assert url == "http://cam01:hz_secret_9921@192.168.10.42:8080/video"

    def test_discover_paired_camera_not_found(self, monkeypatch):
        monkeypatch.delenv("HAZARDLENS_DRONE_URL", raising=False)
        monkeypatch.setattr("utils.security._check_socket_port", lambda host, port, timeout: False)

        url = discover_paired_camera(port=8080, cached_url=None)
        assert url is None


class TestMultiCameraRegistryAndAutoHeal:
    """Tests for config/paired_cameras.json registry, scanning, handshake, and auto-IP healing."""

    def test_registry_save_and_load(self, tmp_path, monkeypatch):
        test_file = str(tmp_path / "paired_test.json")
        monkeypatch.setattr("utils.security.PAIRED_CAMERAS_FILE", test_file)

        assert load_paired_cameras() == {}
        ok = save_camera_slot("CAM 01", "192.168.1.50:8080", user="admin", password="password123")
        assert ok is True

        cfg = get_paired_camera("CAM 01")
        assert cfg is not None
        assert cfg["last_ip_port"] == "192.168.1.50:8080"
        assert cfg["user"] == "admin"
        assert cfg["password"] == "password123"

    def test_verify_camera_handshake_success(self, monkeypatch):
        class MockResponse:
            def getcode(self): return 200
            @property
            def headers(self): return {"Content-Type": "image/jpeg"}
            def read(self, n=16): return b"\xff\xd8\xff\xe0"
            def __enter__(self): return self
            def __exit__(self, *args): pass

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: MockResponse())
        ok = verify_camera_handshake("192.168.1.7:8080", "cam01", "pass")
        assert ok is True

    def test_verify_camera_handshake_rejected_401(self, monkeypatch):
        import urllib.error
        def mock_open(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", mock_open)
        ok = verify_camera_handshake("192.168.1.7:8080", "cam01", "wrong_pass")
        assert ok is False

    def test_verify_camera_handshake_empty_user_sends_auth_none(self, monkeypatch):
        captured_auth = []
        class MockResponse:
            status_code = 200
            content = b"\xff\xd8\xff\xe0"
            headers = {"Content-Type": "image/jpeg"}

        def mock_get(url, auth=None, **kwargs):
            captured_auth.append(auth)
            return MockResponse()

        monkeypatch.setattr("requests.get", mock_get)
        ok = verify_camera_handshake("192.168.1.3:8080", user="", password="")
        assert ok is True
        assert captured_auth == [None]

    def test_scan_network_cameras_discovers_nodes(self, monkeypatch):
        monkeypatch.setattr("utils.security.get_local_subnets", lambda: ["192.168.1.0/24"])
        monkeypatch.setattr("utils.security._check_socket_port", lambda ip, port, timeout: ip in ("192.168.1.7", "192.168.1.20"))

        class MockResponse:
            def getcode(self): return 200
            def __enter__(self): return self
            def __exit__(self, *args): pass

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: MockResponse())
        cams = scan_network_cameras(port=8080)
        assert "192.168.1.7:8080" in cams
        assert "192.168.1.20:8080" in cams

    def test_resolve_paired_camera_fast_path(self, tmp_path, monkeypatch):
        test_file = str(tmp_path / "paired_test.json")
        monkeypatch.setattr("utils.security.PAIRED_CAMERAS_FILE", test_file)
        save_camera_slot("CAM 01", "192.168.1.7:8080", user="cam01", password="secret")

        monkeypatch.setattr("utils.security.verify_camera_handshake", lambda node, user, password, timeout: node == "192.168.1.7:8080")
        url = resolve_paired_camera("CAM 01")
        assert url == "http://cam01:secret@192.168.1.7:8080/video"

    def test_resolve_paired_camera_auto_heal(self, tmp_path, monkeypatch):
        test_file = str(tmp_path / "paired_test.json")
        monkeypatch.setattr("utils.security.PAIRED_CAMERAS_FILE", test_file)
        save_camera_slot("CAM 01", "192.168.1.7:8080", user="cam01", password="secret")

        def mock_handshake(node, user, password, timeout):
            return node == "192.168.137.45:8080" and user == "cam01" and password == "secret"

        monkeypatch.setattr("utils.security.verify_camera_handshake", mock_handshake)
        monkeypatch.setattr("utils.security.scan_network_cameras", lambda timeout: ["192.168.137.45:8080"])

        url = resolve_paired_camera("CAM 01")
        assert url == "http://cam01:secret@192.168.137.45:8080/video"

        updated = get_paired_camera("CAM 01")
        assert updated["last_ip_port"] == "192.168.137.45:8080"


class TestQRPairing:
    """Tests for WhatsApp-Web style zero-input QR pairing."""

    def test_get_laptop_wifi_ip(self):
        from utils.qr_pairing import get_laptop_wifi_ip
        ip = get_laptop_wifi_ip()
        assert isinstance(ip, str)
        assert len(ip.split(".")) == 4

    def test_generate_qr_code_png(self):
        from utils.qr_pairing import generate_qr_code_png
        png_bytes = generate_qr_code_png("http://192.168.1.5:8502/pair?token=test1234")
        assert isinstance(png_bytes, bytes)
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_create_pairing_session_and_status(self):
        from utils.qr_pairing import create_pairing_session, get_pairing_status
        token, url, qr_bytes = create_pairing_session(slot="CAM 01")
        assert len(token) > 10
        assert f"token={token}" in url
        assert qr_bytes.startswith(b"\x89PNG\r\n\x1a\n")

        status = get_pairing_status(token)
        assert status is not None
        assert status["slot"] == "CAM 01"
        assert status["paired"] is False

    def test_pairing_http_handler_simulated(self, tmp_path, monkeypatch):
        from io import BytesIO
        from utils.qr_pairing import (
            PairingHTTPHandler,
            ACTIVE_PAIRING_TOKENS,
            _PAIRING_LOCK,
        )

        test_file = str(tmp_path / "paired_test.json")
        monkeypatch.setattr("utils.security.PAIRED_CAMERAS_FILE", test_file)

        test_token = "mock_token_abc123"
        with _PAIRING_LOCK:
            ACTIVE_PAIRING_TOKENS[test_token] = {
                "slot": "CAM 01",
                "created_at": 0,
                "paired": False,
                "phone_ip": None,
            }

        # Mock socket and request handling
        class MockSocket:
            def __init__(self):
                self.data = BytesIO()
            def makefile(self, mode, *args, **kwargs):
                if "b" in mode:
                    return self.data
                return self.data
            def sendall(self, data):
                self.data.write(data)

        handler = PairingHTTPHandler.__new__(PairingHTTPHandler)
        handler.client_address = ("192.168.1.42", 54321)
        handler.path = f"/pair?token={test_token}&slot=CAM%2001"
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"GET {handler.path} HTTP/1.1"

        handler.do_GET()

        response = handler.wfile.getvalue()
        assert b"200 OK" in response or b"NODE CONNECTED" in response
        assert ACTIVE_PAIRING_TOKENS[test_token]["paired"] is True
        assert ACTIVE_PAIRING_TOKENS[test_token]["phone_ip"] == "192.168.1.42"

        # Check that camera slot was saved
        paired = get_paired_camera("CAM 01")
        assert paired is not None
        assert paired["last_ip_port"] == "192.168.1.42:8080"

    def test_pairing_http_handler_invalid_token(self):
        from io import BytesIO
        from utils.qr_pairing import PairingHTTPHandler

        handler = PairingHTTPHandler.__new__(PairingHTTPHandler)
        handler.client_address = ("192.168.1.99", 54321)
        handler.path = "/pair?token=invalid_nonexistent_token&slot=CAM%2001"
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"GET {handler.path} HTTP/1.1"

        handler.do_GET()
        response = handler.wfile.getvalue()
        assert b"403" in response or b"Invalid or Expired" in response

    def test_qr_session_targets_streamlit_port_8501(self):
        from utils.qr_pairing import create_pairing_session, validate_pairing_token
        token, url, qr_bytes = create_pairing_session("CAM 01")
        assert ":8501" in url
        assert f"pair_token={token}" in url
        assert "slot=CAM01" in url
        assert validate_pairing_token(token) is True
        assert validate_pairing_token("nonexistent_random_token") is False

    def test_mark_pairing_complete(self):
        from utils.qr_pairing import create_pairing_session, get_pairing_status, mark_pairing_complete
        token, _, _ = create_pairing_session("CAM 01")
        status_before = get_pairing_status(token)
        assert status_before["paired"] is False

        mark_pairing_complete(token, "192.168.1.3", "CAM 01")
        status_after = get_pairing_status(token)
        assert status_after["paired"] is True
        assert status_after["phone_ip"] == "192.168.1.3"
