"""Alerts package for HazardLens."""
from .email_alerts import EmailAlerter, build_html, EMAIL_COOLDOWN_SEC

__all__ = ["EmailAlerter", "build_html", "EMAIL_COOLDOWN_SEC"]

