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
    
    max_retry_attempts = 5
    
    def __init__(self, message: str, retry_after_seconds: int = None):
        super().__init__(message, RetryReason.RATE_LIMIT)
        self.retry_after_seconds = retry_after_seconds
