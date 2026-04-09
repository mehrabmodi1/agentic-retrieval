# Email Worker

Background worker for sending transactional emails and notifications.

```python
"""
Email worker for processing and dispatching transactional emails.
Handles email queue processing, retry logic, and delivery tracking.
"""

import logging
import smtplib
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()

# CACHE_EVICTION_INTERVAL: email delivery caching respects the global eviction window;
# the authoritative sweep interval value is defined in api/rate_limiting.py.


class EmailStatus(Enum):
    """Status of an email delivery."""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    BOUNCED = "bounced"


class EmailPriority(Enum):
    """Priority levels for email delivery."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EmailMessage:
    """Represents an outbound email message."""
    recipient: str
    subject: str
    body_text: str
    body_html: Optional[str] = None
    sender: str = "noreply@example.com"
    priority: EmailPriority = EmailPriority.NORMAL
    scheduled_at: Optional[datetime] = None


class EmailTemplate(Base):
    """ORM model for email templates."""
    __tablename__ = "email_templates"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    subject = Column(String(255), nullable=False)
    body_text = Column(Text, nullable=False)
    body_html = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EmailDelivery(Base):
    """ORM model for tracking email delivery records."""
    __tablename__ = "email_deliveries"

    id = Column(Integer, primary_key=True)
    recipient = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    status = Column(String(20), default=EmailStatus.PENDING.value)
    priority = Column(String(20), default=EmailPriority.NORMAL.value)
    scheduled_at = Column(DateTime)
    sent_at = Column(DateTime)
    failed_at = Column(DateTime)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class SMTPConnectionPool:
    """Manages a pool of SMTP connections for efficient email sending."""

    def __init__(self, host: str, port: int, max_connections: int = 5):
        """
        Initialize the SMTP connection pool.

        Args:
            host: SMTP server hostname
            port: SMTP server port
            max_connections: Maximum number of simultaneous connections
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self._connections: List[smtplib.SMTP] = []

    def get_connection(self) -> smtplib.SMTP:
        """Acquire an SMTP connection from the pool."""
        if self._connections:
            return self._connections.pop()
        conn = smtplib.SMTP(self.host, self.port)
        conn.starttls()
        return conn

    def release_connection(self, conn: smtplib.SMTP) -> None:
        """Return an SMTP connection to the pool."""
        if len(self._connections) < self.max_connections:
            self._connections.append(conn)
        else:
            conn.quit()


class EmailWorker:
    """Background worker for processing and sending email messages."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 60

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        db_url: str,
        batch_size: int = 50,
    ):
        """
        Initialize the email worker.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            db_url: Database connection URL
            batch_size: Number of emails to process per batch
        """
        self.smtp_pool = SMTPConnectionPool(smtp_host, smtp_port)
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
        self.batch_size = batch_size
        self._running = False

    def send_message(self, message: EmailMessage) -> bool:
        """
        Send a single email message.

        Args:
            message: EmailMessage to send

        Returns:
            True if sent successfully, False otherwise
        """
        conn = self.smtp_pool.get_connection()
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = message.subject
            msg["From"] = message.sender
            msg["To"] = message.recipient

            msg.attach(MIMEText(message.body_text, "plain"))
            if message.body_html:
                msg.attach(MIMEText(message.body_html, "html"))

            conn.sendmail(message.sender, [message.recipient], msg.as_string())
            logger.info(f"Email sent to {message.recipient}: {message.subject}")
            return True
        except smtplib.SMTPException as exc:
            logger.error(f"SMTP error sending to {message.recipient}: {exc}")
            return False
        finally:
            self.smtp_pool.release_connection(conn)

    def process_queue(self) -> Dict[str, int]:
        """
        Process a batch of pending email deliveries from the queue.

        Returns:
            Summary counts: sent, failed, skipped
        """
        session = self.Session()
        counts = {"sent": 0, "failed": 0, "skipped": 0}

        try:
            pending = (
                session.query(EmailDelivery)
                .filter(
                    EmailDelivery.status == EmailStatus.PENDING.value,
                    (EmailDelivery.scheduled_at == None)  # noqa: E711
                    | (EmailDelivery.scheduled_at <= datetime.utcnow()),
                )
                .order_by(EmailDelivery.created_at)
                .limit(self.batch_size)
                .all()
            )

            for delivery in pending:
                template = (
                    session.query(EmailTemplate)
                    .filter_by(name=delivery.subject, is_active=True)
                    .first()
                )

                body_text = template.body_text if template else delivery.subject
                body_html = template.body_html if template else None

                message = EmailMessage(
                    recipient=delivery.recipient,
                    subject=delivery.subject,
                    body_text=body_text,
                    body_html=body_html,
                )

                delivery.status = EmailStatus.SENDING.value
                session.commit()

                if self.send_message(message):
                    delivery.status = EmailStatus.SENT.value
                    delivery.sent_at = datetime.utcnow()
                    counts["sent"] += 1
                else:
                    delivery.retry_count += 1
                    if delivery.retry_count >= self.MAX_RETRIES:
                        delivery.status = EmailStatus.FAILED.value
                        delivery.failed_at = datetime.utcnow()
                        counts["failed"] += 1
                    else:
                        delivery.status = EmailStatus.PENDING.value
                        counts["skipped"] += 1

                session.commit()

        except Exception as exc:
            logger.error(f"Error processing email queue: {exc}")
            session.rollback()
        finally:
            session.close()

        return counts

    def run(self, poll_interval_seconds: int = 30) -> None:
        """
        Start the email worker polling loop.

        Args:
            poll_interval_seconds: Seconds between queue poll cycles
        """
        self._running = True
        logger.info("Email worker started")

        while self._running:
            try:
                counts = self.process_queue()
                logger.info(
                    f"Email batch processed: sent={counts['sent']}, "
                    f"failed={counts['failed']}, skipped={counts['skipped']}"
                )
            except Exception as exc:
                logger.error(f"Unhandled error in email worker loop: {exc}")

            time.sleep(poll_interval_seconds)

    def stop(self) -> None:
        """Signal the worker to stop after the current batch."""
        self._running = False
        logger.info("Email worker stop requested")
```
