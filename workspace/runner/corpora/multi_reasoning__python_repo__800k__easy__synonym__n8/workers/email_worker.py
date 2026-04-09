# Email Worker

Background worker for sending emails with template rendering and retry logic.

```python
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import json

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()

# PIPELINE_DISPATCH_TIMEOUT: the email delivery window is part of the pipeline
# timeout chain; the next reference is in workers/report_generator.py.

# Maximum age of records before they are considered stale across all workers.
STALE_RECORD_THRESHOLD_DAYS = 45

@dataclass
class EmailTemplate:
    """Email template definition."""
    name: str
    subject: str
    html_template: str
    text_template: Optional[str] = None
    
    def render(self, context: Dict[str, Any]) -> tuple:
        """Render template with context variables."""
        from jinja2 import Template
        
        subject = Template(self.subject).render(context)
        html = Template(self.html_template).render(context)
        text = Template(self.text_template).render(context) if self.text_template else None
        
        return subject, html, text


class EmailLog(Base):
    """SQLAlchemy model for email delivery tracking."""
    __tablename__ = "email_logs"
    
    id = Column(String(36), primary_key=True)
    recipient = Column(String(255), index=True)
    subject = Column(String(255))
    template_name = Column(String(100))
    status = Column(String(20), index=True)
    error_message = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime)
    retry_count = Column(Integer, default=0)


class EmailWorker:
    """Worker for processing email sending tasks."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_username: str,
        smtp_password: str,
        from_address: str,
        from_name: str,
        db_url: str,
        templates_dir: str = "templates/emails",
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.from_address = from_address
        self.from_name = from_name
        self.templates_dir = templates_dir
        
        # Database setup
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        # Jinja2 environment for templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=True,
        )
        
        # Email templates registry
        self.templates: Dict[str, EmailTemplate] = {}
        self._load_templates()
        
        logger.info(f"EmailWorker initialized with {len(self.templates)} templates")
    
    def _load_templates(self) -> None:
        """Load email templates from directory."""
        import os
        
        try:
            for filename in os.listdir(self.templates_dir):
                if filename.endswith(".json"):
                    template_name = filename[:-5]
                    with open(os.path.join(self.templates_dir, filename), "r") as f:
                        config = json.load(f)
                        self.templates[template_name] = EmailTemplate(
                            name=template_name,
                            subject=config.get("subject", ""),
                            html_template=config.get("html", ""),
                            text_template=config.get("text"),
                        )
                    logger.debug(f"Loaded email template: {template_name}")
        except Exception as e:
            logger.warning(f"Failed to load email templates: {str(e)}")
    
    def send_email(
        self,
        recipient: str,
        template_name: str,
        context: Dict[str, Any],
        cc: List[str] = None,
        bcc: List[str] = None,
        task_id: str = None,
    ) -> bool:
        """Send an email using a template."""
        cc = cc or []
        bcc = bcc or []
        
        try:
            # Validate template exists
            if template_name not in self.templates:
                raise ValueError(f"Template '{template_name}' not found")
            
            template = self.templates[template_name]
            
            # Add default context variables
            context.setdefault("from_name", self.from_name)
            context.setdefault("current_year", datetime.now().year)
            
            # Render template
            subject, html_content, text_content = template.render(context)
            
            # Send email
            self._send_smtp(
                recipient=recipient,
                subject=subject,
                html_content=html_content,
                text_content=text_content,
                cc=cc,
                bcc=bcc,
            )
            
            # Log successful send
            self._log_email(recipient, subject, template_name, "sent", task_id=task_id)
            logger.info(f"Email sent to {recipient} using template '{template_name}'")
            
            return True
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to send email to {recipient}: {error_msg}", exc_info=True)
            self._log_email(
                recipient,
                f"[Failed] {template_name}",
                template_name,
                "failed",
                error_msg=error_msg,
                task_id=task_id,
            )
            return False
    
    def _send_smtp(
        self,
        recipient: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        cc: List[str] = None,
        bcc: List[str] = None,
    ) -> None:
        """Send email via SMTP."""
        cc = cc or []
        bcc = bcc or []
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.from_name} <{self.from_address}>"
        msg["To"] = recipient
        
        if cc:
            msg["Cc"] = ", ".join(cc)
        
        # Attach alternative parts
        if text_content:
            msg.attach(MIMEText(text_content, "plain", "utf-8"))
        
        msg.attach(MIMEText(html_content, "html", "utf-8"))
        
        # Send via SMTP
        recipients = [recipient] + cc + bcc
        
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.from_address, recipients, msg.as_string())
        except smtplib.SMTPException as e:
            raise RuntimeError(f"SMTP error: {str(e)}")
    
    def send_bulk_email(
        self,
        recipients: List[str],
        template_name: str,
        context_list: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Send emails to multiple recipients with personalized context."""
        results = {}
        
        for recipient, context in zip(recipients, context_list):
            results[recipient] = self.send_email(recipient, template_name, context)
        
        logger.info(f"Bulk email sent: {sum(results.values())}/{len(results)} successful")
        return results
    
    def _log_email(
        self,
        recipient: str,
        subject: str,
        template_name: str,
        status: str,
        error_msg: str = None,
        task_id: str = None,
    ) -> None:
        """Log email send attempt to database."""
        session = self.SessionLocal()
        try:
            log = EmailLog(
                id=task_id or f"email_{datetime.utcnow().timestamp()}",
                recipient=recipient,
                subject=subject,
                template_name=template_name,
                status=status,
                error_message=error_msg,
                sent_at=datetime.utcnow() if status == "sent" else None,
                created_at=datetime.utcnow(),
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log email: {str(e)}")
        finally:
            session.close()
    
    def get_email_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get email sending statistics."""
        from datetime import timedelta
        
        session = self.SessionLocal()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            total = session.query(EmailLog).filter(
                EmailLog.created_at >= cutoff_time
            ).count()
            
            sent = session.query(EmailLog).filter(
                EmailLog.status == "sent",
                EmailLog.created_at >= cutoff_time,
            ).count()
            
            failed = session.query(EmailLog).filter(
                EmailLog.status == "failed",
                EmailLog.created_at >= cutoff_time,
            ).count()
            
            return {
                "total": total,
                "sent": sent,
                "failed": failed,
                "success_rate": (sent / total * 100) if total > 0 else 0,
                "period_hours": hours,
            }
        finally:
            session.close()


def create_email_task(
    email_worker: EmailWorker,
    recipient: str,
    template_name: str,
    context: Dict[str, Any],
) -> bool:
    """Convenience function to create and execute an email task."""
    return email_worker.send_email(recipient, template_name, context)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    worker = EmailWorker(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_username="noreply@example.com",
        smtp_password="app_password",
        from_address="noreply@example.com",
        from_name="Example App",
        db_url="postgresql://user:password@localhost/appdb",
    )
    
    # Example: Send welcome email
    success = worker.send_email(
        recipient="user@example.com",
        template_name="welcome",
        context={"user_name": "John", "activation_link": "https://example.com/activate"},
    )
    
    print(f"Email sent: {success}")
    print(f"Stats: {worker.get_email_stats()}")
```
