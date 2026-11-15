"""
Check that the Resend email setup works on this machine.

Usage:
    python scripts/check_resend.py          # only checks config
    python scripts/check_resend.py --send   # also sends one real test email
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = ["RESEND_API_KEY", "HAZARDLENS_FROM_EMAIL", "HAZARDLENS_ADMIN_EMAIL"]


def check_config() -> bool:
    ok = True
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if not value:
            print(f"[FAIL] {var} is not set")
            ok = False
        else:
            print(f"[ OK ] {var} is set")

    key = os.getenv("RESEND_API_KEY", "")
    if key and not key.startswith("re_"):
        print("[WARN] RESEND_API_KEY does not start with 're_', check the key")
        ok = False

    return ok


def send_test_email() -> bool:
    import resend

    resend.api_key = os.getenv("RESEND_API_KEY")
    try:
        result = resend.Emails.send({
            "from": os.getenv("HAZARDLENS_FROM_EMAIL"),
            "to": [os.getenv("HAZARDLENS_ADMIN_EMAIL")],
            "subject": "HazardLens: Resend test",
            "html": "<p>Resend is working for HazardLens.</p>",
        })
        print(f"[ OK ] Test email accepted by Resend, id: {result['id']}")
        return True
    except Exception as exc:
        print(f"[FAIL] Resend rejected the request: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Resend setup for HazardLens")
    parser.add_argument("--send", action="store_true", help="send one real test email")
    args = parser.parse_args()

    if not check_config():
        print("\nFix the .env values (see .env.example) and run again.")
        return 1

    if not args.send:
        print("\nConfig looks good. Run with --send to send a real test email.")
        return 0

    return 0 if send_test_email() else 1


if __name__ == "__main__":
    sys.exit(main())

