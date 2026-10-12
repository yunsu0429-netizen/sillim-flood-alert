"""Gmail SMTP 발송. 계정이 없으면 data/outbox/에 .eml로 남기고 경고 로그."""
import logging
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage

from pathlib import Path

from src.config import settings

log = logging.getLogger(__name__)
OUTBOX = Path(settings.db_path).parent / "outbox"


def configured() -> bool:
    return bool(settings.gmail_user and settings.gmail_app_password)


def _send(msg: EmailMessage, force_outbox: bool = False) -> str:
    if force_outbox or not configured():
        OUTBOX.mkdir(parents=True, exist_ok=True)
        path = OUTBOX / f"{datetime.now():%Y%m%d-%H%M%S-%f}.eml"
        path.write_bytes(bytes(msg))
        log.warning("Gmail 미설정 → 메일을 보내지 않고 %s 에 저장 (받는 사람 %s)", path.name, msg["To"])
        return "outbox"
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as s:
        s.starttls()
        s.login(settings.gmail_user, settings.gmail_app_password)
        s.send_message(msg)
    log.info("메일 발송: %s / %s", msg["To"], msg["Subject"])
    return "gmail"


def send(to: str, subject: str, text: str, html: str | None = None, background=True, force_outbox=False) -> str:
    msg = EmailMessage()
    msg["From"] = settings.gmail_user or "sillim-flood-alert@localhost"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    channel = "gmail" if configured() and not force_outbox else "outbox"
    if background:
        def run():
            try:
                _send(msg, force_outbox)
            except Exception:
                log.exception("메일 발송 실패: %s", to)
        threading.Thread(target=run, daemon=True).start()
        return channel
    return _send(msg, force_outbox)
