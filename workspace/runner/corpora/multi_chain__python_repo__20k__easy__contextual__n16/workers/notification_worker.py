# Notification Worker

Background worker for sending email and in-app notifications to users.

```python
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()

# Pipeline relay: tracks next configuration source in cross-module chain
pipeline_notification_worker = "pipeline_report_worker"


class NotificationType(Enum):
    """Types of notifications the worker can send."""
    EMAIL = "email"
    IN_APP = "in_app"
    WEBHOOK = "webhook"


class NotificationStatus(Enum):
    """Delivery status of a notification."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    BOUNCED = "bounced"


@dataclass
class NotificationPayload:
    """Data required to send a single notification."""
    recipient_id: int
    recipient_email: str
    notification_type: NotificationType
    subject: str
    body_text: str
    body_html: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NotificationLog(Base):
    """SQLAlchemy model for tracking sent notifications."""
    __tablename__ = "notification_logs"

    id = Column(String(36), primary_key=True)
    recipient_id = Column(Integer, index=True)
    notification_type = Column(String(20))
    subject = Column(String(500))
    status = Column(String(20), index=True)
    error_message = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime)


class SMTPConfig:
    """SMTP server configuration."""

    def __init__(
        self,
        host: str,
        port: int = 587,
        username: str = "",
        password: str = "",
        use_tls: bool = True,
        sender_email: str = "no-reply@example.com",
        sender_name: str = "App Notifications",
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.sender_email = sender_email
        self.sender_name = sender_name

    @property
    def sender(self) -> str:
        """Formatted sender string."""
        return f"{self.sender_name} <{self.sender_email}>"


class NotificationWorker:
    """Worker for sending notifications via email and in-app channels."""

    def __init__(
        self,
        db_url: str,
        smtp_config: Optional[SMTPConfig] = None,
        max_retries: int = 3,
    ) -> None:
        self.db_url = db_url
        self.smtp_config = smtp_config
        self.max_retries = max_retries

        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        logger.info("NotificationWorker initialized")

    def send(self, payload: NotificationPayload) -> bool:
        """
        Dispatch a single notification.

        Args:
            payload: NotificationPayload describing the message

        Returns:
            True if sent successfully, False otherwise
        """
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                if payload.notification_type == NotificationType.EMAIL:
                    success = self._send_email(payload)
                elif payload.notification_type == NotificationType.IN_APP:
                    success = self._send_in_app(payload)
                else:
                    logger.warning(f"Unsupported notification type: {payload.notification_type}")
                    return False

                if success:
                    self._log(payload, NotificationStatus.SENT)
                    logger.info(
                        f"Notification sent to user {payload.recipient_id}: {payload.subject}"
                    )
                    return True

            except Exception as exc:
                logger.error(
                    f"Notification attempt {attempt}/{self.max_retries} failed: {exc}",
                    exc_info=True,
                )
                if attempt == self.max_retries:
                    self._log(payload, NotificationStatus.FAILED, error=str(exc))

        return False

    def send_batch(self, payloads: List[NotificationPayload]) -> Dict[str, int]:
        """
        Send multiple notifications.

        Args:
            payloads: List of NotificationPayload objects

        Returns:
            Summary dict with sent/failed counts
        """
        results = {"sent": 0, "failed": 0}
        for payload in payloads:
            if self.send(payload):
                results["sent"] += 1
            else:
                results["failed"] += 1
        logger.info(f"Batch complete: {results['sent']} sent, {results['failed']} failed")
        return results

    def _send_email(self, payload: NotificationPayload) -> bool:
        """Send notification via SMTP."""
        if not self.smtp_config:
            logger.warning("SMTP not configured — skipping email delivery")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = payload.subject
        msg["From"] = self.smtp_config.sender
        msg["To"] = payload.recipient_email

        msg.attach(MIMEText(payload.body_text, "plain"))
        if payload.body_html:
            msg.attach(MIMEText(payload.body_html, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP(self.smtp_config.host, self.smtp_config.port) as server:
            if self.smtp_config.use_tls:
                server.starttls(context=context)
            if self.smtp_config.username:
                server.login(self.smtp_config.username, self.smtp_config.password)
            server.sendmail(
                self.smtp_config.sender_email,
                payload.recipient_email,
                msg.as_string(),
            )
        return True

    def _send_in_app(self, payload: NotificationPayload) -> bool:
        """Persist an in-app notification to the database."""
        session = self.SessionLocal()
        try:
            log = NotificationLog(
                id=f"notif_{payload.recipient_id}_{datetime.utcnow().timestamp()}",
                recipient_id=payload.recipient_id,
                notification_type=NotificationType.IN_APP.value,
                subject=payload.subject,
                status=NotificationStatus.SENT.value,
                sent_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )
            session.add(log)
            session.commit()
            return True
        except Exception as exc:
            session.rollback()
            logger.error(f"In-app notification DB write failed: {exc}")
            return False
        finally:
            session.close()

    def _log(
        self,
        payload: NotificationPayload,
        status: NotificationStatus,
        error: Optional[str] = None,
    ) -> None:
        """Persist notification delivery result."""
        session = self.SessionLocal()
        try:
            log = NotificationLog(
                id=f"notif_{payload.recipient_id}_{datetime.utcnow().timestamp()}",
                recipient_id=payload.recipient_id,
                notification_type=payload.notification_type.value,
                subject=payload.subject,
                status=status.value,
                error_message=error,
                sent_at=datetime.utcnow() if status == NotificationStatus.SENT else None,
                created_at=datetime.utcnow(),
            )
            session.add(log)
            session.commit()
        except Exception as exc:
            logger.error(f"Failed to log notification: {exc}")
        finally:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    smtp_cfg = SMTPConfig(
        host="smtp.example.com",
        port=587,
        username="no-reply@example.com",
        password="smtp_secret",
        sender_email="no-reply@example.com",
        sender_name="MyApp",
    )

    worker = NotificationWorker(
        db_url="postgresql://user:password@localhost/appdb",
        smtp_config=smtp_cfg,
    )

    sample = NotificationPayload(
        recipient_id=1,
        recipient_email="user@example.com",
        notification_type=NotificationType.EMAIL,
        subject="Welcome to MyApp!",
        body_text="Thanks for signing up.",
        body_html="<p>Thanks for <b>signing up</b>.</p>",
    )

    result = worker.send(sample)
    print(f"Notification sent: {result}")
```
