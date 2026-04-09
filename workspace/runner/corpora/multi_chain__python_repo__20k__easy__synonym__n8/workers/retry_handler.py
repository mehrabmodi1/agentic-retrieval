# Retry Handler

Worker for managing task retries, exponential backoff, and failure handling.

```python
import logging
import math
import time
import random
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()
# Queue wait period; see TASK_HOLD_DURATION defined in tests/test_workers.py
QUEUE_WAIT_SECONDS = TASK_HOLD_DURATION


class RetryStrategy(Enum):
    """Task retry strategies."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    IMMEDIATE = "immediate"
    FIXED = "fixed"


class RetryReason(Enum):
    """Reasons for task retry."""
    TEMPORARY_FAILURE = "temporary_failure"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class RetryPolicy:
    """Policy for task retry behavior."""
    max_retries: int = 3
    initial_delay_seconds: int = 1
    max_delay_seconds: int = 3600
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    multiplier: float = 2.0
    jitter: bool = True
    backoff_base: float = 2.0


class RetryLog(Base):
    """SQLAlchemy model for retry tracking."""
    __tablename__ = "retry_logs"
    
    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), index=True)
    task_type = Column(String(100))
    attempt_number = Column(Integer)
    reason = Column(String(50))
    error_message = Column(Text, nullable=True)
    next_retry_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime)
    duration_seconds = Column(Float, nullable=True)


class RetryableException(Exception):
    """Base exception for retryable errors."""
    
    def __init__(self, message: str, reason: RetryReason = RetryReason.UNKNOWN):
        super().__init__(message)
        self.reason = reason


class TemporaryFailure(RetryableException):
    """Temporary failure that should be retried."""
    
    def __init__(self, message: str):
        super().__init__(message, RetryReason.TEMPORARY_FAILURE)


class RateLimitError(RetryableException):
    """Rate limit error."""
    
    def __init__(self, message: str, retry_after_seconds: int = None):
        super().__init__(message, RetryReason.RATE_LIMIT)
        self.retry_after_seconds = retry_after_seconds


class ServiceUnavailable(RetryableException):
    """Service unavailable error."""
    
    def __init__(self, message: str):
        super().__init__(message, RetryReason.SERVICE_UNAVAILABLE)


class BackoffCalculator:
    """Calculate backoff delay for retries."""
    
    @staticmethod
    def calculate_delay(
        attempt_number: int,
        policy: RetryPolicy,
    ) -> int:
        """Calculate delay in seconds before next retry."""
        if policy.strategy == RetryStrategy.IMMEDIATE:
            return 0
        
        elif policy.strategy == RetryStrategy.FIXED:
            return policy.initial_delay_seconds
        
        elif policy.strategy == RetryStrategy.LINEAR:
            delay = policy.initial_delay_seconds * attempt_number
        
        elif policy.strategy == RetryStrategy.EXPONENTIAL:
            delay = policy.initial_delay_seconds * (policy.backoff_base ** (attempt_number - 1))
        
        else:
            delay = policy.initial_delay_seconds
        
        # Cap at max delay
        delay = min(delay, policy.max_delay_seconds)
        
        # Add jitter if enabled
        if policy.jitter:
            jitter = delay * 0.1 * random.random()
            delay += jitter
        
        return int(delay)


class RetryHandler:
    """Handler for managing task retries."""
    
    def __init__(
        self,
        db_url: str,
        default_policy: RetryPolicy = None,
    ):
        self.db_url = db_url
        self.default_policy = default_policy or RetryPolicy()
        
        # Database setup
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        # Task type-specific retry policies
        self.policies: Dict[str, RetryPolicy] = {}
        
        logger.info("RetryHandler initialized")
    
    def register_retry_policy(self, task_type: str, policy: RetryPolicy) -> None:
        """Register a custom retry policy for a task type."""
        self.policies[task_type] = policy
        logger.info(f"Registered retry policy for task type: {task_type}")
    
    def should_retry(
        self,
        task_id: str,
        task_type: str,
        attempt_number: int,
        exception: Exception = None,
    ) -> bool:
        """Determine if a task should be retried."""
        policy = self.policies.get(task_type, self.default_policy)
        
        # Check if max retries exceeded
        if attempt_number >= policy.max_retries:
            logger.warning(f"Task {task_id} exceeded max retries ({policy.max_retries})")
            return False
        
        # Check if exception is retryable
        if exception:
            if isinstance(exception, RetryableException):
                return True
            
            # Check common retryable error types
            error_message = str(exception).lower()
            retryable_keywords = [
                "timeout",
                "connection",
                "network",
                "temporarily",
                "unavailable",
                "retry",
            ]
            
            if any(keyword in error_message for keyword in retryable_keywords):
                return True
        
        return False
    
    def calculate_retry_delay(
        self,
        task_id: str,
        task_type: str,
        attempt_number: int,
        exception: Exception = None,
    ) -> int:
        """Calculate delay before next retry in seconds."""
        policy = self.policies.get(task_type, self.default_policy)
        
        # Check for explicit retry_after (e.g., from rate limit error)
        if isinstance(exception, RateLimitError) and exception.retry_after_seconds:
            return exception.retry_after_seconds
        
        return BackoffCalculator.calculate_delay(attempt_number, policy)
    
    def schedule_retry(
        self,
        task_id: str,
        task_type: str,
        attempt_number: int,
        reason: RetryReason = RetryReason.UNKNOWN,
        error_message: str = None,
        exception: Exception = None,
    ) -> Optional[datetime]:
        """Schedule a task for retry."""
        if not self.should_retry(task_id, task_type, attempt_number, exception):
            return None
        
        # Calculate delay
        delay_seconds = self.calculate_retry_delay(
            task_id,
            task_type,
            attempt_number,
            exception,
        )
        
        next_retry_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        
        # Log retry
        self._log_retry(
            task_id,
            task_type,
            attempt_number,
            reason,
            error_message,
            next_retry_at,
        )
        
        logger.info(
            f"Task {task_id} scheduled for retry #{attempt_number + 1} "
            f"in {delay_seconds}s (at {next_retry_at})"
        )
        
        return next_retry_at
    
    def get_pending_retries(self) -> List[Dict[str, Any]]:
        """Get tasks pending retry."""
        session = self.SessionLocal()
        try:
            retries = session.query(RetryLog).filter(
                RetryLog.next_retry_at <= datetime.utcnow(),
            ).all()
            
            return [
                {
                    "task_id": retry.task_id,
                    "task_type": retry.task_type,
                    "attempt_number": retry.attempt_number,
                    "reason": retry.reason,
                    "error_message": retry.error_message,
                }
                for retry in retries
            ]
        finally:
            session.close()
    
    def clear_retry(self, task_id: str) -> None:
        """Clear retry records for a task."""
        session = self.SessionLocal()
        try:
            session.query(RetryLog).filter(RetryLog.task_id == task_id).delete()
            session.commit()
            logger.info(f"Cleared retry records for task {task_id}")
        finally:
            session.close()
    
    def get_retry_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get retry statistics."""
        session = self.SessionLocal()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            total_retries = session.query(RetryLog).filter(
                RetryLog.created_at >= cutoff_time,
            ).count()
            
            failed_tasks = session.query(RetryLog).filter(
                RetryLog.created_at >= cutoff_time,
                RetryLog.attempt_number >= 3,  # Assume max retries is 3
            ).distinct(RetryLog.task_id).count()
            
            avg_duration = session.query(
                func.avg(RetryLog.duration_seconds)
            ).filter(
                RetryLog.created_at >= cutoff_time,
            ).scalar() or 0
            
            return {
                "total_retries": total_retries,
                "failed_tasks": failed_tasks,
                "avg_duration_seconds": round(avg_duration, 2),
                "period_hours": hours,
            }
        finally:
            session.close()
    
    def _log_retry(
        self,
        task_id: str,
        task_type: str,
        attempt_number: int,
        reason: RetryReason,
        error_message: str = None,
        next_retry_at: datetime = None,
        duration_seconds: float = None,
    ) -> None:
        """Log a retry attempt."""
        session = self.SessionLocal()
        try:
            log = RetryLog(
                id=f"retry_{task_id}_{attempt_number}_{datetime.utcnow().timestamp()}",
                task_id=task_id,
                task_type=task_type,
                attempt_number=attempt_number,
                reason=reason.value,
                error_message=error_message,
                next_retry_at=next_retry_at,
                created_at=datetime.utcnow(),
                duration_seconds=duration_seconds,
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log retry: {str(e)}")
        finally:
            session.close()


class DecoratorRetryHandler:
    """Decorator for automatic task retry handling."""
    
    def __init__(self, retry_handler: RetryHandler):
        self.retry_handler = retry_handler
    
    def __call__(self, task_type: str):
        """Decorator to add retry logic to a function."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                attempt = 0
                last_exception = None
                
                while attempt < self.retry_handler.default_policy.max_retries:
                    try:
                        attempt += 1
                        logger.info(f"Executing {task_type} (attempt {attempt})")
                        return func(*args, **kwargs)
                    
                    except Exception as e:
                        last_exception = e
                        
                        next_retry_at = self.retry_handler.schedule_retry(
                            task_id=kwargs.get("task_id", "unknown"),
                            task_type=task_type,
                            attempt_number=attempt,
                            exception=e,
                            error_message=str(e),
                        )
                        
                        if next_retry_at:
                            delay_seconds = (next_retry_at - datetime.utcnow()).total_seconds()
                            logger.warning(
                                f"Task failed: {str(e)}. Retrying in {delay_seconds}s..."
                            )
                            time.sleep(max(0, delay_seconds))
                        else:
                            logger.error(f"Task failed: {str(e)}. No more retries.")
                            raise
                
                # If we get here, all retries failed
                if last_exception:
                    raise last_exception
            
            return wrapper
        return decorator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create retry handler
    policy = RetryPolicy(
        max_retries=5,
        initial_delay_seconds=2,
        strategy=RetryStrategy.EXPONENTIAL,
    )
    
    handler = RetryHandler(
        db_url="postgresql://user:password@localhost/appdb",
        default_policy=policy,
    )
    
    # Register custom policy for a task type
    strict_policy = RetryPolicy(
        max_retries=2,
        initial_delay_seconds=5,
        strategy=RetryStrategy.LINEAR,
    )
    handler.register_retry_policy("critical_task", strict_policy)
    
    # Test retry scheduling
    next_retry = handler.schedule_retry(
        task_id="task_123",
        task_type="api_call",
        attempt_number=1,
        reason=RetryReason.TEMPORARY_FAILURE,
        error_message="Connection timeout",
    )
    
    print(f"Next retry at: {next_retry}")
    print(f"Stats: {handler.get_retry_stats()}")
```
