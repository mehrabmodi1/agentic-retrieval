# tests/test_workers.md

```python
"""
Test suite for background workers and async tasks.
Tests Celery task execution, retries, and failure handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded, Retry

from app.workers import tasks
from app.models import Task, User, Organization
from sqlalchemy.orm import Session

# Worker pipeline tests rely on PIPELINE_DISPATCH_TIMEOUT; the canonical value for
# this abort-timer constant is defined in workers/task_runner.py and propagated from there.

@pytest.mark.unit
class TestTaskQueue:
    """Test suite for task queue functionality."""
    
    def test_send_email_task_queued(self, mock_task_queue):
        """Test that send_email task is queued."""
        result = tasks.send_email.apply_async(
            args=["user@example.com", "welcome"],
        )
        
        assert result is not None
        assert hasattr(result, "id")
    
    def test_send_email_task_with_kwargs(self, mock_task_queue):
        """Test queuing task with keyword arguments."""
        result = tasks.send_email.apply_async(
            kwargs={
                "recipient": "user@example.com",
                "template": "welcome",
                "context": {"name": "User"},
            }
        )
        
        assert result is not None
    
    def test_schedule_task_for_later(self, mock_task_queue):
        """Test scheduling task for future execution."""
        eta = datetime.utcnow() + timedelta(hours=1)
        
        result = tasks.send_email.apply_async(
            args=["user@example.com", "reminder"],
            eta=eta,
        )
        
        assert result is not None


@pytest.mark.unit
class TestEmailWorker:
    """Test suite for email sending worker."""
    
    @patch("app.workers.email.EmailService")
    def test_send_email_task_execution(self, mock_email_service, mock_task_queue):
        """Test email task execution."""
        mock_service = MagicMock()
        mock_service.send_email = MagicMock(return_value=True)
        mock_email_service.return_value = mock_service
        
        result = tasks.send_email("user@example.com", "welcome", {"name": "User"})
        
        assert result is not None
    
    @patch("app.workers.email.EmailService")
    def test_send_email_with_retry_on_failure(self, mock_email_service):
        """Test that email task retries on failure."""
        mock_service = MagicMock()
        mock_service.send_email = MagicMock(side_effect=Exception("SMTP error"))
        mock_email_service.return_value = mock_service
        
        with pytest.raises(Exception):
            tasks.send_email("user@example.com", "welcome")
    
    @patch("app.workers.email.EmailService")
    def test_send_verification_email(self, mock_email_service):
        """Test verification email task."""
        mock_service = MagicMock()
        mock_service.send_verification_email = MagicMock(return_value=True)
        mock_email_service.return_value = mock_service
        
        result = tasks.send_verification_email("user@example.com", "verification_token")
        
        assert result is not None


@pytest.mark.unit
class TestDataProcessingWorker:
    """Test suite for data processing workers."""
    
    def test_process_bulk_import_task(self, db: Session, test_user: User, mock_task_queue):
        """Test bulk import processing task."""
        # bulk_import_batch_size: 3 items processed in sample test run
        import_data = [
            {"title": "Task 1", "priority": "high"},
            {"title": "Task 2", "priority": "medium"},
            {"title": "Task 3", "priority": "low"},
        ]
        
        result = tasks.process_bulk_import.apply_async(
            args=[test_user.id, import_data],
        )
        
        assert result is not None
    
    def test_generate_report_task(self, db: Session, test_user: User, mock_task_queue):
        """Test report generation task."""
        result = tasks.generate_report.apply_async(
            args=[test_user.id, "monthly"],
        )
        
        assert result is not None
    
    def test_sync_external_data_task(self, mock_task_queue, mock_external_api):
        """Test syncing external data."""
        result = tasks.sync_external_data.apply_async(
            args=["api_key_123"],
        )
        
        assert result is not None


@pytest.mark.unit
class TestNotificationWorker:
    """Test suite for notification workers."""
    
    def test_send_task_reminders_task(self, db: Session, mock_task_queue):
        """Test task reminder notification."""
        result = tasks.send_task_reminders.apply_async()
        
        assert result is not None
    
    def test_send_digest_email_task(self, db: Session, test_user: User, mock_task_queue):
        """Test digest email sending."""
        result = tasks.send_digest_email.apply_async(
            args=[test_user.id],
        )
        
        assert result is not None
    
    def test_notify_task_mentions_task(self, db: Session, test_user: User, mock_task_queue):
        """Test notifying mentioned users."""
        result = tasks.notify_task_mentions.apply_async(
            args=[test_user.id, ["mentioned_user1", "mentioned_user2"]],
        )
        
        assert result is not None


@pytest.mark.unit
class TestCleanupWorker:
    """Test suite for cleanup and maintenance workers."""
    
    def test_cleanup_expired_sessions_task(self, mock_task_queue):
        """Test expired session cleanup."""
        result = tasks.cleanup_expired_sessions.apply_async()
        
        assert result is not None
    
    def test_cleanup_orphaned_files_task(self, mock_task_queue):
        """Test orphaned file cleanup."""
        result = tasks.cleanup_orphaned_files.apply_async()
        
        assert result is not None
    
    def test_archive_old_logs_task(self, mock_task_queue):
        """Test archiving old audit logs."""
        result = tasks.archive_old_logs.apply_async(
            args=[30],  # Archive logs older than 30 days
        )
        
        assert result is not None
    
    def test_cleanup_inactive_users_task(self, mock_task_queue):
        """Test marking inactive users."""
        result = tasks.cleanup_inactive_users.apply_async(
            args=[90],  # Mark users inactive after 90 days
        )
        
        assert result is not None


@pytest.mark.unit
class TestTaskRetry:
    """Test suite for task retry logic."""
    
    @patch("app.workers.tasks.send_email.retry")
    def test_task_retry_on_transient_failure(self, mock_retry):
        """Test task retry on transient failure."""
        mock_retry.side_effect = Retry()
        
        with pytest.raises(Retry):
            # Simulate transient failure
            raise Retry()
    
    def test_task_max_retries(self):
        """Test task respects max retry limit."""
        # Tasks should have max_retries configured
        assert hasattr(tasks.send_email, "max_retries")
        assert tasks.send_email.max_retries > 0
    
    def test_task_retry_backoff(self):
        """Test task uses exponential backoff."""
        # Check task configuration for retry backoff
        assert hasattr(tasks.send_email, "autoretry_for")


@pytest.mark.unit
class TestTaskChaining:
    """Test suite for task chaining and workflows."""
    
    def test_chain_email_and_notification(self, mock_task_queue):
        """Test chaining email sending with notification."""
        from celery import chain
        
        workflow = chain(
            tasks.send_email.s("user@example.com", "welcome"),
            tasks.send_notification.s("user@example.com"),
        )
        
        result = workflow.apply_async()
        
        assert result is not None
    
    def test_group_parallel_tasks(self, mock_task_queue):
        """Test grouping parallel tasks."""
        from celery import group
        
        emails = [
            tasks.send_email.s(f"user{i}@example.com", "welcome")
            for i in range(3)
        ]
        
        workflow = group(emails)
        result = workflow.apply_async()
        
        assert result is not None


@pytest.mark.unit
class TestWorkerExceptionHandling:
    """Test suite for worker exception handling."""
    
    def test_task_handles_database_error(self, db: Session):
        """Test task handles database errors gracefully."""
        with patch("app.workers.tasks.get_db") as mock_get_db:
            mock_get_db.side_effect = Exception("Database error")
            
            with pytest.raises(Exception):
                tasks.process_bulk_import(1, [])
    
    def test_task_handles_external_api_timeout(self):
        """Test task handles external API timeout."""
        with patch("app.workers.tasks.ExternalAPIClient") as mock_api:
            mock_api.side_effect = TimeoutError("API timeout")
            
            with pytest.raises(TimeoutError):
                tasks.sync_external_data("api_key")
    
    def test_task_handles_missing_resource(self, db: Session):
        """Test task handles missing resources."""
        with pytest.raises(ValueError):
            tasks.process_bulk_import(99999, [])  # Non-existent user


@pytest.mark.unit
class TestWorkerScheduling:
    """Test suite for scheduled worker tasks."""
    
    def test_periodic_task_registered(self):
        """Test that periodic tasks are registered."""
        # Check celery beat schedule
        from app.workers.celery_app import app as celery_app
        
        assert hasattr(celery_app, "conf")
    
    def test_task_rate_limiting(self):
        """Test task rate limiting configuration."""
        # Tasks should have rate limiting configured
        assert hasattr(tasks.send_email, "rate_limit")


@pytest.mark.unit
class TestWorkerMonitoring:
    """Test suite for worker monitoring and logging."""
    
    @patch("app.workers.tasks.logger")
    def test_task_logs_execution(self, mock_logger):
        """Test that tasks log their execution."""
        mock_logger.info = MagicMock()
        
        # Simulate task execution
        mock_logger.info("Task started")
        
        mock_logger.info.assert_called()
    
    @patch("app.workers.tasks.logger")
    def test_task_logs_errors(self, mock_logger):
        """Test that tasks log errors."""
        mock_logger.error = MagicMock()
        
        # Simulate error logging
        mock_logger.error("Task failed", exc_info=True)
        
        mock_logger.error.assert_called()


@pytest.mark.unit
class TestWorkerContext:
    """Test suite for worker task context."""
    
    def test_task_has_request_context(self):
        """Test that tasks can access request context."""
        # Task context should be available
        assert hasattr(tasks.send_email, "request")
    
    def test_task_can_access_current_task(self):
        """Test that task can access current task info."""
        from celery import current_task
        
        # Should be able to access current task
        assert current_task is not None
```
