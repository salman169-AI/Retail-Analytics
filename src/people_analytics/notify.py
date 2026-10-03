"""Alert notifications by email (SMTP), free with an ordinary mail account.

Gmail, Outlook and most providers accept SMTP from an "app password", so no paid
service is involved (SMS gateways such as Twilio charge per message). The
password is never kept in the config: it is read from the environment variable
named in `notify.email.password_env`.

Every alert is also written to `notify.outbox_dir` as an .eml file, whether or
not email is enabled, so there's a record of exactly what was (or would have
been) sent. Open one in any mail client to preview it.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from loguru import logger

from people_analytics.config import NotifyConfig


def build_email(cfg: NotifyConfig, subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.email.sender or cfg.email.username or "alerts@localhost"
    msg["To"] = ", ".join(cfg.email.to) or "manager@localhost"
    msg["Date"] = datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M:%S %z")
    msg.set_content(body)
    return msg


def send(cfg: NotifyConfig, subject: str, body: str, tag: str = "alert") -> dict:
    """Write the alert to the outbox and, if enabled, email it. Returns what happened."""
    msg = build_email(cfg, subject, body)
    out = Path(cfg.outbox_dir)
    out.mkdir(parents=True, exist_ok=True)
    eml = out / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}.eml"
    eml.write_bytes(bytes(msg))
    result = {"outbox": str(eml), "emailed": False}

    e = cfg.email
    if not e.enabled:
        return result
    password = os.environ.get(e.password_env)
    if not (e.username and password and e.to):
        logger.warning(f"Email alert not sent: set notify.email.username/to and the "
                       f"{e.password_env} environment variable.")
        return result
    try:
        with smtplib.SMTP(e.smtp_host, e.smtp_port, timeout=20) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(e.username, password)
            smtp.send_message(msg)
        result["emailed"] = True
        logger.info(f"Alert emailed to {', '.join(e.to)}")
    except (OSError, smtplib.SMTPException) as err:  # network or auth failure
        logger.error(f"Email alert failed: {err}")
        result["error"] = str(err)
    return result


def alert_text(site: str, message: str, clip_time_s: float, source: str) -> tuple[str, str]:
    m, s = divmod(int(clip_time_s), 60)
    subject = f"[{site}] Counter unattended: {message.split(' waiting')[0]} waiting"
    body = (f"{message}\n\n"
            f"Site: {site}\n"
            f"Camera: {source}\n"
            f"Video time: {m:02d}:{s:02d}\n")
    return subject, body
