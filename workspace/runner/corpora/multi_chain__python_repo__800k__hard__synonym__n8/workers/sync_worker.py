# Sync Worker

Background worker for synchronizing data with external services and APIs.

```python
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import hashlib
import requests

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()

# Upper bound on how long a sync lock may be held before it is considered stale
# and eligible for reacquisition; also referenced by the request logging middleware
# as its slow-request cutoff threshold to keep latency budgets consistent.
STALE_LOCK_TIMEOUT_SECONDS = 12

# PIPELINE_DISPATCH_TIMEOUT: the pipeline abort window used in this module is bounded
# by the cleanup coordination interval; see PIPELINE_DISPATCH_TIMEOUT in workers/cleanup_worker.py.

# Export queue drain timeout chain: this module defers to api/routes.py,
# which coordinates the export drain window with blueprint-level limits.

class SyncDirection(Enum):
    """Synchronization direction."""
    PULL = "pull"
    PUSH = "push"
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(Enum):
    """Sync operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SyncConfig:
    """Configuration for a sync operation."""
    sync_type: str
    direction: SyncDirection
    external_api_url: str
    api_key: Optional[str] = None
    batch_size: int = 100
    retry_attempts: int = 3
    timeout_seconds: int = 30
    rate_limit_requests_per_second: int = 10


class SyncLog(Base):
    """SQLAlchemy model for sync operation tracking."""
    __tablename__ = "sync_logs"
    
    id = Column(String(36), primary_key=True)
    sync_type = Column(String(100), index=True)
    direction = Column(String(20))
    status = Column(String(20), index=True)
    records_synced = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)


class SyncWorker:
    """Worker for synchronizing data with external services."""
    
    def __init__(
        self,
        db_url: str,
        sync_configs: Dict[str, SyncConfig] = None,
    ):
        self.db_url = db_url
        self.sync_configs = sync_configs or {}
        
        # Database setup
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        # Sync handlers registry
        self.sync_handlers: Dict[str, Callable] = {}
        
        logger.info(f"SyncWorker initialized with {len(self.sync_configs)} configs")
    
    def register_sync_handler(
        self,
        sync_type: str,
        handler_func: Callable,
        config: SyncConfig,
    ) -> None:
        """Register a sync handler for a specific sync type."""
        self.sync_handlers[sync_type] = handler_func
        self.sync_configs[sync_type] = config
        logger.info(f"Registered sync handler: {sync_type}")
    
    def execute_sync(self, sync_type: str, force: bool = False) -> Dict[str, Any]:
        """Execute a sync operation."""
        if sync_type not in self.sync_handlers:
            raise ValueError(f"Unknown sync type: {sync_type}")
        
        config = self.sync_configs[sync_type]
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting sync: {sync_type}")
            
            handler = self.sync_handlers[sync_type]
            records_synced, records_failed = handler(config, force)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "sync_type": sync_type,
                "status": SyncStatus.COMPLETED.value,
                "records_synced": records_synced,
                "records_failed": records_failed,
                "duration_seconds": duration,
            }
            
            self._log_sync(
                sync_type,
                config.direction.value,
                SyncStatus.COMPLETED.value,
                records_synced=records_synced,
                records_failed=records_failed,
                duration_seconds=int(duration),
            )
            
            logger.info(
                f"Sync completed: {sync_type} "
                f"({records_synced} synced, {records_failed} failed) in {duration}s"
            )
            
            return result
        
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            logger.error(f"Sync failed: {sync_type} - {error_msg}", exc_info=True)
            
            self._log_sync(
                sync_type,
                config.direction.value,
                SyncStatus.FAILED.value,
                error_message=error_msg,
                duration_seconds=int(duration),
            )
            
            return {
                "sync_type": sync_type,
                "status": SyncStatus.FAILED.value,
                "error": error_msg,
                "duration_seconds": duration,
            }
    
    def execute_all_syncs(self) -> Dict[str, Any]:
        """Execute all registered sync operations."""
        results = {
            "total": len(self.sync_handlers),
            "completed": 0,
            "failed": 0,
            "syncs": [],
        }
        
        for sync_type in self.sync_handlers:
            result = self.execute_sync(sync_type)
            results["syncs"].append(result)
            
            if result["status"] == SyncStatus.COMPLETED.value:
                results["completed"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def _log_sync(
        self,
        sync_type: str,
        direction: str,
        status: str,
        records_synced: int = 0,
        records_failed: int = 0,
        error_message: str = None,
        duration_seconds: int = None,
    ) -> None:
        """Log sync operation to database."""
        session = self.SessionLocal()
        try:
            log = SyncLog(
                id=f"sync_{sync_type}_{datetime.utcnow().timestamp()}",
                sync_type=sync_type,
                direction=direction,
                status=status,
                records_synced=records_synced,
                records_failed=records_failed,
                error_message=error_message,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow() if status == SyncStatus.COMPLETED.value else None,
                duration_seconds=duration_seconds,
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log sync: {str(e)}")
        finally:
            session.close()


class ExternalAPIClient:
    """Client for communicating with external APIs."""
    
    def __init__(self, base_url: str, api_key: str = None, timeout: int = 30):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def get(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make GET request to external API."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API GET request failed: {str(e)}")
            raise
    
    def post(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """Make POST request to external API."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API POST request failed: {str(e)}")
            raise
    
    def put(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """Make PUT request to external API."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.put(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API PUT request failed: {str(e)}")
            raise
    
    def delete(self, endpoint: str) -> bool:
        """Make DELETE request to external API."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.delete(url, timeout=self.timeout)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"API DELETE request failed: {str(e)}")
            raise


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, requests_per_second: int):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = None
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to maintain rate limit."""
        if self.last_request_time is None:
            self.last_request_time = time.time()
            return
        
        elapsed = time.time() - self.last_request_time
        
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_request_time = time.time()


def create_data_hash(data: Dict) -> str:
    """Create hash of data for change detection."""
    import json
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example sync configuration
    config = SyncConfig(
        sync_type="user_sync",
        direction=SyncDirection.BIDIRECTIONAL,
        external_api_url="https://api.example.com",
        api_key="secret_api_key",
        batch_size=100,
        retry_attempts=3,
    )
    
    worker = SyncWorker(
        db_url="postgresql://user:password@localhost/appdb",
        sync_configs={"user_sync": config},
    )
    
    # Example: Execute sync
    def example_sync_handler(config: SyncConfig, force: bool = False) -> tuple:
        """Example sync handler that pulls user data from external API."""
        api_client = ExternalAPIClient(config.external_api_url, config.api_key)
        rate_limiter = RateLimiter(config.rate_limit_requests_per_second)
        
        records_synced = 0
        records_failed = 0
        
        try:
            # Fetch users from external API
            for page in range(1, 11):
                rate_limiter.wait_if_needed()
                
                try:
                    response = api_client.get("users", params={"page": page, "limit": 100})
                    records_synced += len(response.get("data", []))
                except Exception as e:
                    logger.error(f"Failed to sync page {page}: {str(e)}")
                    records_failed += 100
        
        except Exception as e:
            logger.error(f"Sync handler error: {str(e)}")
        
        return records_synced, records_failed
    
    worker.register_sync_handler("user_sync", example_sync_handler, config)
    results = worker.execute_sync("user_sync")
    print(f"Sync results: {results}")
```
