"""
Email service for HumanGuard — sends verification emails via AWS SES.

Falls back to logging the verification link when SES is not configured
(no SENDER_EMAIL env var or boto3 unavailable), so local dev keeps working.
"""

import logging
import os

logger = logging.getLogger(__name__)

VERIFY_BASE_URL = "https://d1hi33wespusty.cloudfront.net/verify.html"

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Verify your HumanGuard API key</title>
<style>
  body {{ margin:0; padding:0; background:#0d1117; font-family:system-ui,sans-serif; }}
  .wrap {{ max-width:560px; margin:48px auto; background:#161b22; border:1px solid #30363d;
           border-radius:12px; overflow:hidden; }}
  .header {{ background:#21262d; padding:32px; text-align:center; }}
  .logo {{ font-size:22px; font-weight:700; color:#e6edf3; letter-spacing:1px; }}
  .logo span {{ color:#3fb950; }}
  .body {{ padding:32px; }}
  h2 {{ color:#e6edf3; margin:0 0 16px; font-size:20px; }}
  p {{ color:#8b949e; line-height:1.6; margin:0 0 20px; font-size:14px; }}
  .btn {{ display:inline-block; background:#238636; color:#fff; text-decoration:none;
          padding:12px 28px; border-radius:8px; font-size:14px; font-weight:600; }}
  .mono {{ background:#0d1117; color:#79c0ff; padding:12px 16px; border-radius:6px;
           font-family:monospace; font-size:13px; word-break:break-all;
           border:1px solid #30363d; margin-bottom:20px; }}
  .footer {{ padding:20px 32px; border-top:1px solid #30363d; }}
  .footer p {{ font-size:12px; color:#484f58; margin:0; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="logo">Human<span>Guard</span></div>
  </div>
  <div class="body">
    <h2>Verify your API key</h2>
    <p>Thanks for registering! Click the button below to verify your email address
       and activate your HumanGuard API key. This link expires in <strong>24 hours</strong>.</p>
    <p style="text-align:center">
      <a class="btn" href="{verify_url}">Verify my email</a>
    </p>
    <p>Or copy this link into your browser:</p>
    <div class="mono">{verify_url}</div>
    <p>If you did not request an API key, you can safely ignore this email.</p>
  </div>
  <div class="footer">
    <p>HumanGuard &mdash; Bot detection for the modern web</p>
  </div>
</div>
</body>
</html>
"""


def send_verification_email(email: str, token: str, api_key_id: str) -> bool:
    """Send a verification email via AWS SES.

    Returns True if sent (or logged as fallback), False on hard failure.

    Args:
        email:      Recipient address.
        token:      The opaque verification token.
        api_key_id: The non-secret key_id portion (e.g. 'hg_live_abc12345').
    """
    verify_url = f"{VERIFY_BASE_URL}?token={token}&id={api_key_id}"

    sender = os.environ.get("SENDER_EMAIL", "")
    if not sender:
        # No SES configured — log the link so local dev can still test.
        logger.info(
            "SES not configured (SENDER_EMAIL unset). Verification link for %s: %s",
            email,
            verify_url,
        )
        return True

    html_body = _HTML_TEMPLATE.format(verify_url=verify_url)
    text_body = (
        f"Verify your HumanGuard API key\n\n"
        f"Open this link to verify your email address (expires in 24 hours):\n"
        f"{verify_url}\n\n"
        f"If you did not request an API key, ignore this email."
    )

    try:
        import boto3

        ses = boto3.client("ses", region_name="us-east-1")
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": [email]},
            Message={
                "Subject": {"Data": "Verify your HumanGuard API key", "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": text_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                },
            },
        )
        logger.info("Verification email sent to %s (key_id=%s)", email, api_key_id)
        return True
    except Exception as exc:
        logger.warning(
            "Failed to send verification email to %s via SES: %s — link: %s",
            email,
            exc,
            verify_url,
        )
        return False
