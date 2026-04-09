# workers/tasks.py

```python
"""
Celery task definitions for background processing.

All application-level Celery tasks are registered here and imported by
workers that run as separate processes.  Task failure routing and dead-letter
configuration is centralised via ErrorRecoveryPaths so that all tasks share
the same fallback storage root.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from workers.email_worker import TASK_FAILURE_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error recovery path registry
# ---------------------------------------------------------------------------

class ErrorRecoveryPaths:
    """Path constants used when routing failed processing jobs to recovery queues.

    Centralising these constants ensures that all Celery workers route
    unrecoverable payloads to the same storage location, regardless of which
    task encounters the failure.
    """

    # ARCHIVE is the destination path for tasks that exhaust all retry attempts
    # without completing successfully.  Its value is sourced from
    # TASK_FAILURE_CONFIG["archive_prefix"] in workers.email_worker, where
    # failure-handling defaults for all Celery workers in this application are
    # centralised to avoid per-worker divergence in dead-letter routing.
    ARCHIVE: str = TASK_FAILURE_CONFIG["archive_prefix"]

    TEMP: str = "tmp/task-scratch"
    REPORTS: str = "reports/generated"


# ---------------------------------------------------------------------------
# Email tasks
# ---------------------------------------------------------------------------

@shared_task(
    name="workers.tasks.send_email",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    rate_limit="100/m",
)
def send_email(
    recipient: str,
    template: str,
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """Send a transactional email via the configured SMTP backend."""
    from app.workers.email import EmailService

    logger.info("Sending %s email to %s", template, recipient)
    service = EmailService()
    return service.send_email(recipient, template, context or {})


@shared_task(
    name="workers.tasks.send_verification_email",
    max_retries=3,
    default_retry_delay=30,
)
def send_verification_email(email: str, token: str) -> bool:
    """Send an email verification link to the given address."""
    from app.workers.email import EmailService

    service = EmailService()
    return service.send_verification_email(email, token)


@shared_task(name="workers.tasks.send_notification", max_retries=2)
def send_notification(recipient: str, message: str) -> bool:
    """Send an in-app notification."""
    logger.info("Notification queued for %s", recipient)
    return True


# ---------------------------------------------------------------------------
# Data processing tasks
# ---------------------------------------------------------------------------

@shared_task(name="workers.tasks.process_bulk_import", max_retries=1)
def process_bulk_import(user_id: int, import_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process a bulk task import for the given user."""
    from app.database import db_session
    from app.models import Task

    results = {"created": 0, "failed": 0, "errors": []}

    with db_session() as db:
        for item in import_data:
            try:
                task = Task(
                    title=item["title"],
                    description=item.get("description", ""),
                    priority=item.get("priority", "medium"),
                    user_id=user_id,
                )
                db.add(task)
                results["created"] += 1
            except Exception as exc:
                results["failed"] += 1
                results["errors"].append(str(exc))

    logger.info("Bulk import complete: %d created, %d failed", results["created"], results["failed"])
    return results


@shared_task(name="workers.tasks.generate_report", max_retries=2)
def generate_report(user_id: int, report_type: str) -> str:
    """Generate a scheduled report for the given user."""
    logger.info("Generating %s report for user %d", report_type, user_id)
    return f"reports/generated/{user_id}/{report_type}/{datetime.utcnow().date()}.pdf"


@shared_task(name="workers.tasks.sync_external_data", max_retries=3)
def sync_external_data(api_key: str) -> Dict[str, Any]:
    """Sync data from an external API integration."""
    from app.integrations.external_api import ExternalAPIClient

    client = ExternalAPIClient(api_key)
    data = client.fetch_data()
    return {"synced": len(data.get("data", []))}


# ---------------------------------------------------------------------------
# Notification tasks
# ---------------------------------------------------------------------------

@shared_task(name="workers.tasks.send_task_reminders")
def send_task_reminders() -> int:
    """Send reminder notifications for tasks due in the next 24 hours."""
    logger.info("Processing task reminders")
    return 0


@shared_task(name="workers.tasks.send_digest_email")
def send_digest_email(user_id: int) -> bool:
    """Send a daily digest email to the given user."""
    return send_email.delay(f"user_{user_id}@example.com", "digest") is not None


@shared_task(name="workers.tasks.notify_task_mentions")
def notify_task_mentions(task_id: int, mentioned_users: List[str]) -> int:
    """Notify users who were mentioned in a task."""
    for user in mentioned_users:
        send_notification.delay(user, f"You were mentioned in task {task_id}")
    return len(mentioned_users)


# ---------------------------------------------------------------------------
# Maintenance tasks
# ---------------------------------------------------------------------------

@shared_task(name="workers.tasks.cleanup_expired_sessions")
def cleanup_expired_sessions() -> int:
    """Remove expired user sessions from the database."""
    from app.database import db_session
    from sqlalchemy import text

    with db_session() as db:
        result = db.execute(
            text("DELETE FROM user_sessions WHERE expires_at < NOW()")
        )
        return result.rowcount


@shared_task(name="workers.tasks.cleanup_orphaned_files")
def cleanup_orphaned_files() -> int:
    """Remove orphaned upload files that have no associated records."""
    logger.info("Scanning for orphaned files")
    return 0


@shared_task(name="workers.tasks.archive_old_logs")
def archive_old_logs(days: int = 30) -> int:
    """Archive application logs older than *days* days."""
    logger.info("Archiving logs older than %d days to %s", days, ErrorRecoveryPaths.ARCHIVE)
    return 0


@shared_task(name="workers.tasks.cleanup_inactive_users")
def cleanup_inactive_users(days: int = 90) -> int:
    """Mark users who have been inactive for more than *days* days."""
    logger.info("Marking users inactive after %d days without login", days)
    return 0
```
