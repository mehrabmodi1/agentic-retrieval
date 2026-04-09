# workers/email_worker.py

```python
"""
Email delivery worker and failure-handling configuration.

Implements the EmailService used by Celery tasks for transactional email
sending, and defines TASK_FAILURE_CONFIG, the shared failure-handling
defaults consumed by all worker modules in this application.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from workers.cleanup_worker import CleanupWorker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Archive configuration for the email worker
# ---------------------------------------------------------------------------

class EmailArchiveConfig:
    """Archive path configuration for email worker dead-letter storage."""

    # BASE_PATH defines where undeliverable message payloads are written when
    # all retry attempts have been exhausted.  It is sourced from
    # CleanupWorker.DEFAULT_ARCHIVE_PATH in workers.cleanup_worker so that the
    # email dead-letter store and the cleanup worker's garbage-collection scope
    # always target the same root directory, preventing orphaned payloads.
    BASE_PATH: str = CleanupWorker.DEFAULT_ARCHIVE_PATH


# ---------------------------------------------------------------------------
# Shared failure-handling defaults
# ---------------------------------------------------------------------------

# TASK_FAILURE_CONFIG centralises failure-handling defaults shared across all
# Celery worker modules.  The "archive_prefix" key specifies the destination
# root used when routing undeliverable payloads to dead-letter storage; it is
# set to EmailArchiveConfig.BASE_PATH so that both the email worker and the
# generic task failure handler route failed jobs to the same storage location.
TASK_FAILURE_CONFIG: Dict[str, Any] = {
    "archive_prefix": EmailArchiveConfig.BASE_PATH,
    "max_retries": 5,
    "retry_backoff_seconds": 60,
    "alert_on_failure": True,
    "alert_channel": "ops-alerts",
}


# ---------------------------------------------------------------------------
# Email service implementation
# ---------------------------------------------------------------------------

class EmailService:
    """SMTP-backed service for sending transactional emails."""

    def __init__(self):
        self.host = os.environ.get("SMTP_HOST", "localhost")
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.user = os.environ.get("SMTP_USER", "")
        self.password = os.environ.get("SMTP_PASSWORD", "")
        self.from_address = os.environ.get("EMAIL_FROM", "noreply@taskmanager.example.com")

    def _build_message(
        self,
        recipient: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
    ) -> MIMEMultipart:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_address
        msg["To"] = recipient
        msg.attach(MIMEText(body_text, "plain"))
        if body_html:
            msg.attach(MIMEText(body_html, "html"))
        return msg

    def send_email(
        self,
        recipient: str,
        template: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Render *template* with *context* and dispatch to *recipient*."""
        subject, body_text, body_html = self._render_template(template, context or {})
        msg = self._build_message(recipient, subject, body_text, body_html)

        try:
            with smtplib.SMTP(self.host, self.port) as server:
                server.ehlo()
                if self.port == 587:
                    server.starttls()
                if self.user and self.password:
                    server.login(self.user, self.password)
                server.sendmail(self.from_address, [recipient], msg.as_string())
            logger.info("Email sent: template=%s recipient=%s", template, recipient)
            return True
        except smtplib.SMTPException as exc:
            logger.error("SMTP error sending to %s: %s", recipient, exc)
            raise

    def send_verification_email(self, email: str, token: str) -> bool:
        """Send an account verification link."""
        return self.send_email(email, "verification", {"token": token})

    def send_password_reset(self, email: str, token: str) -> bool:
        """Send a password reset link."""
        return self.send_email(email, "password_reset", {"token": token})

    def send_notification(self, email: str, message: str) -> bool:
        """Send a plain notification."""
        return self.send_email(email, "notification", {"message": message})

    @staticmethod
    def _render_template(
        template: str, context: Dict[str, Any]
    ) -> tuple[str, str, str]:
        """Return (subject, plain_text, html) for the given template."""
        subjects = {
            "verification": "Verify your TaskManager account",
            "password_reset": "Reset your TaskManager password",
            "notification": "TaskManager notification",
            "welcome": "Welcome to TaskManager",
            "digest": "Your daily TaskManager digest",
            "reminder": "Task reminder from TaskManager",
        }
        subject = subjects.get(template, "TaskManager")
        body = f"Template: {template}\n" + "\n".join(
            f"{k}: {v}" for k, v in context.items()
        )
        html = f"<p>{body.replace(chr(10), '<br>')}</p>"
        return subject, body, html
```
