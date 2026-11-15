import os
import time
from unittest.mock import patch
import pytest
import resend
from alerts.email_alerts import EmailAlerter, build_html, EMAIL_COOLDOWN_SEC


def test_cooldown_blocks_second_send_within_120s():
    """Requirement (a): Cooldown blocks a second send within the cooldown window."""
    with patch("resend.Emails.send", return_value={"id": "test_msg_id"}):
        alerter = EmailAlerter(
            api_key="re_test_key_12345",
            from_email="onboarding@resend.dev",
            admin_email="admin@example.com",
            cooldown_sec=120,
            backoff_delays=(0.01, 0.02, 0.04),
        )

        try:
            assert alerter.is_enabled is True

            # 1. First send for FIRE should succeed
            assert alerter.send_alert("FIRE") is True

            # 2. Second send for FIRE within 120s should be blocked by cooldown
            assert alerter.send_alert("FIRE") is False

            # 3. Different hazard (CROWD) has its own cooldown window and should succeed
            assert alerter.send_alert("CROWD", person_count=6) is True

            # 4. Immediate second send for CROWD should be blocked
            assert alerter.send_alert("CROWD", person_count=7) is False

            # 5. Advance time past 120s for FIRE and verify it can send again
            with patch("time.time", return_value=time.time() + 125):
                assert alerter.send_alert("FIRE") is True

        finally:
            alerter.shutdown(timeout=1.0)


def test_retry_happens_on_exception():
    """Requirement (b): Retry happens with exponential backoff on exceptions."""
    with patch("resend.Emails.send") as mock_send, patch("time.sleep") as mock_sleep:
        # Fails 2 times, succeeds on 3rd attempt
        mock_send.side_effect = [
            RuntimeError("Network failure 1"),
            RuntimeError("Network failure 2"),
            {"id": "re_success_789"},
        ]

        alerter = EmailAlerter(
            api_key="re_test_key_12345",
            from_email="onboarding@resend.dev",
            admin_email="admin@example.com",
            backoff_delays=(1.0, 2.0, 4.0),
        )

        try:
            success = alerter._process_send("FIRE", 0, "2026-11-16 00:00:00")
            assert success is True
            assert mock_send.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)
        finally:
            alerter.shutdown(timeout=1.0)

    # Test that persistent failure exhausts all retries (4 total attempts)
    with patch("resend.Emails.send") as mock_send_fail, patch("time.sleep") as mock_sleep_fail:
        mock_send_fail.side_effect = ConnectionError("Permanent connection error")

        alerter_fail = EmailAlerter(
            api_key="re_test_key_12345",
            from_email="onboarding@resend.dev",
            admin_email="admin@example.com",
            backoff_delays=(1.0, 2.0, 4.0),
        )

        try:
            success = alerter_fail._process_send("CROWD", 12, "2026-11-16 00:00:00")
            assert success is False
            assert mock_send_fail.call_count == 4
            assert mock_sleep_fail.call_count == 3
            mock_sleep_fail.assert_any_call(1.0)
            mock_sleep_fail.assert_any_call(2.0)
            mock_sleep_fail.assert_any_call(4.0)
        finally:
            alerter_fail.shutdown(timeout=1.0)


def test_missing_env_vars_disable_sending_without_raising():
    """Requirement (c): Missing environment variables disable sending cleanly without raising."""
    with patch("alerts.email_alerts.load_dotenv"), patch.dict(os.environ, {}, clear=True):
        alerter = EmailAlerter(api_key=None, from_email=None, admin_email=None)

        assert alerter.is_enabled is False
        assert alerter.worker_thread is None

        assert alerter.send_alert("FIRE") is False
        assert alerter.send_alert("CROWD", person_count=10) is False

        alerter.shutdown(timeout=0.1)

    alerter_empty = EmailAlerter(api_key="", from_email="", admin_email="")
    assert alerter_empty.is_enabled is False
    assert alerter_empty.worker_thread is None
    assert alerter_empty.send_alert("FIRE") is False


def test_build_html_template():
    """Verify HTML template structure and color coding."""
    fire_html = build_html("FIRE", 0, "2026-11-16 00:00:00")
    assert "#FF3A3A" in fire_html
    assert "#240000" in fire_html
    assert "FIRE" in fire_html
    assert "Fire Safety Team required immediately!" in fire_html

    crowd_html = build_html("CROWD", 8, "2026-11-16 00:00:00")
    assert "#FFB800" in crowd_html
    assert "#291c00" in crowd_html
    assert "CROWD" in crowd_html
    assert "Live Count: 8 persons" in crowd_html

