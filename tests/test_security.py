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
    DEFAULT_BEST_PT_SHA256,
)


class TestStreamUrlValidation:
    """Tests for SSRF and Local File Inclusion defense."""

    @pytest.mark.parametrize("valid_url", [
        "http://192.168.1.7:8080/video",
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
