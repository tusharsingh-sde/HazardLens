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
    normalize_camera_url,
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
        "https://hazardlens-cam.trycloudflare.com",
        "https://sub.trycloudflare.com/shot.jpg",
        "hazardlens-cam.trycloudflare.com",
        "192.168.1.3:8080",
    ])
    def test_valid_stream_urls_accepted(self, valid_url):
        ok, reason = validate_stream_url(valid_url)
        assert ok is True
        assert len(reason) > 0

    def test_normalize_camera_url(self):
        assert normalize_camera_url("http://192.168.1.3:8080/video") == "http://192.168.1.3:8080"
        assert normalize_camera_url("http://192.168.1.3:8080/shot.jpg") == "http://192.168.1.3:8080"
        assert normalize_camera_url("192.168.1.3:8080") == "http://192.168.1.3:8080"
        assert normalize_camera_url("https://xyz.trycloudflare.com/shot.jpg") == "https://xyz.trycloudflare.com"
        assert normalize_camera_url("xyz.trycloudflare.com") == "https://xyz.trycloudflare.com"
        assert normalize_camera_url("https://xyz.trycloudflare.com/video/") == "https://xyz.trycloudflare.com"

        # Appending /shot.jpg produces no double slashes
        base = normalize_camera_url("https://xyz.trycloudflare.com/video")
        assert f"{base}/shot.jpg" == "https://xyz.trycloudflare.com/shot.jpg"

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
        from email.message import Message
        def mock_open(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", Message(), None)

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
        assert updated is not None and isinstance(updated, dict)
        assert updated["last_ip_port"] == "192.168.137.45:8080"


class TestSecurityHelpers:
    """Tests for security helper utilities."""

    def test_clean_ip_address_strips_ipv6_mapped_and_ports(self):
        from utils.security import clean_ip_address
        assert clean_ip_address("::ffff:192.168.1.14") == "192.168.1.14"
        assert clean_ip_address("::ffff:192.168.1.3:8080") == "192.168.1.3"
        assert clean_ip_address("http://192.168.1.3:8080/") == "192.168.1.3"
        assert clean_ip_address("[::ffff:192.168.1.14]:8501") == "192.168.1.14"
        assert clean_ip_address("127.0.0.1") == "127.0.0.1"
        assert clean_ip_address(None) is None
        assert clean_ip_address("") is None


    def test_save_camera_slot_rejects_laptop_own_ip_and_loopback(self, tmp_path, monkeypatch):
        from utils.security import save_camera_slot, get_paired_camera
        test_file = str(tmp_path / "test_reg.json")
        monkeypatch.setattr("utils.security.PAIRED_CAMERAS_FILE", test_file)
        monkeypatch.setattr("utils.security.get_local_ip", lambda: "192.168.1.14")

        # Saving laptop IP or loopback MUST return False and not save
        assert save_camera_slot("CAM 01", "192.168.1.14:8080") is False
        assert save_camera_slot("CAM 01", "::ffff:192.168.1.14:8080") is False
        assert save_camera_slot("CAM 01", "127.0.0.1:8080") is False
        assert save_camera_slot("CAM 01", "localhost:8080") is False

        # Saving genuine camera IP MUST succeed
        assert save_camera_slot("CAM 01", "192.168.1.3:8080") is True
        cam = get_paired_camera("CAM 01")
        assert cam is not None and isinstance(cam, dict)
        assert cam["last_ip_port"] == "192.168.1.3:8080"

    def test_discover_cameras_on_lan_filters_local_ips(self, monkeypatch):
        from utils.security import discover_cameras_on_lan
        monkeypatch.setattr("utils.security.get_local_ip", lambda: "192.168.1.14")
        mock_scan = ["192.168.1.14:8080", "127.0.0.1:8080", "192.168.1.3:8080", "::ffff:192.168.1.14:8080"]
        monkeypatch.setattr("utils.security.scan_network_cameras", lambda port, timeout: mock_scan)

        discovered = discover_cameras_on_lan(port=8080)
        assert discovered == ["192.168.1.3:8080"]
