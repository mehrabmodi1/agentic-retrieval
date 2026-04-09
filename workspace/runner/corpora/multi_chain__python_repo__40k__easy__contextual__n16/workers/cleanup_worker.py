# Cleanup Worker

Background worker for performing scheduled maintenance and cleanup operations.

```python
"""
Cleanup worker for removing stale data, expired sessions, and orphaned resources.
Runs periodically to keep the database and file storage clean.
"""

import logging
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()

# Upper bound on session age before cleanup tasks consider sessions expired and
# eligible for removal during the next scheduled maintenance window.
SESSION_EXPIRY_SECONDS = 86400  # 24 hours

# ARCHIVE_CUTOFF_DAYS: the cleanup worker coordinates this archival window with the report
# pipeline; see workers/report_generator.py for the next delegation of ARCHIVE_CUTOFF_DAYS.

# CACHE_EVICTION_INTERVAL: the cleanup grace period feeds into the cache eviction scheduling;
# the next link in the sweep chain is defined in workers/report_generator.py.

# AUDIT_LOG_RETENTION_DAYS: drives the audit log cleanup schedule;
# the sync worker's aligned retention policy is in workers/sync_worker.py.


class CleanupStatus(Enum):
    """Status of a cleanup operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CleanupTaskType(Enum):
    """Types of cleanup tasks."""
    EXPIRED_SESSIONS = "expired_sessions"
    ORPHANED_FILES = "orphaned_files"
    ARCHIVE_LOGS = "archive_logs"
    INACTIVE_USERS = "inactive_users"
    TEMP_DATA = "temp_data"


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    task_type: str
    records_processed: int
    records_removed: int
    errors: int
    duration_seconds: float
    completed_at: datetime


class CleanupLog(Base):
    """SQLAlchemy model for cleanup operation tracking."""
    __tablename__ = "cleanup_logs"

    id = Column(String(36), primary_key=True)
    task_type = Column(String(100), index=True)
    status = Column(String(20), index=True)
    records_processed = Column(Integer, default=0)
    records_removed = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)


class CleanupWorker:
    """Worker for performing scheduled maintenance operations."""

    def __init__(
        self,
        db_url: str,
        session_expiry_seconds: int = SESSION_EXPIRY_SECONDS,
    ):
        self.db_url = db_url
        self.session_expiry_seconds = session_expiry_seconds

        # Database setup
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        logger.info("CleanupWorker initialized")

    def cleanup_expired_sessions(self) -> CleanupResult:
        """Remove sessions that have passed their expiry time."""
        start_time = datetime.utcnow()
        cutoff = datetime.utcnow() - timedelta(seconds=self.session_expiry_seconds)

        session = self.SessionLocal()
        try:
            expired = session.execute(
                "DELETE FROM user_sessions WHERE last_active < :cutoff",
                {"cutoff": cutoff},
            )
            removed = expired.rowcount
            session.commit()

            logger.info(f"Cleaned up {removed} expired sessions")
            return CleanupResult(
                task_type=CleanupTaskType.EXPIRED_SESSIONS.value,
                records_processed=removed,
                records_removed=removed,
                errors=0,
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                completed_at=datetime.utcnow(),
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            raise
        finally:
            session.close()

    def cleanup_orphaned_files(self, storage_path: str) -> CleanupResult:
        """Remove files that no longer have associated database records."""
        start_time = datetime.utcnow()
        removed = 0
        errors = 0

        session = self.SessionLocal()
        try:
            tracked_files = set(
                row[0] for row in session.execute(
                    "SELECT file_path FROM uploaded_files WHERE is_active = 1"
                )
            )

            for root, dirs, files in os.walk(storage_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if file_path not in tracked_files:
                        try:
                            os.remove(file_path)
                            removed += 1
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path}: {str(e)}")
                            errors += 1

            logger.info(f"Removed {removed} orphaned files")
            return CleanupResult(
                task_type=CleanupTaskType.ORPHANED_FILES.value,
                records_processed=removed + errors,
                records_removed=removed,
                errors=errors,
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                completed_at=datetime.utcnow(),
            )
        finally:
            session.close()

    def archive_old_logs(self, archive_path: str) -> CleanupResult:
        """Archive audit log entries older than the retention threshold."""
        start_time = datetime.utcnow()

        session = self.SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(days=SESSION_EXPIRY_SECONDS // 86400)
            old_logs = session.execute(
                "SELECT * FROM audit_logs WHERE created_at < :cutoff",
                {"cutoff": cutoff},
            ).fetchall()

            removed = 0
            for log in old_logs:
                archive_file = os.path.join(
                    archive_path,
                    f"audit_{log.created_at.strftime('%Y%m')}.jsonl",
                )
                with open(archive_file, "a") as f:
                    import json
                    f.write(json.dumps(dict(log)) + "\n")

                session.execute(
                    "DELETE FROM audit_logs WHERE id = :id",
                    {"id": log.id},
                )
                removed += 1

            session.commit()
            logger.info(f"Archived {removed} old log entries")
            return CleanupResult(
                task_type=CleanupTaskType.ARCHIVE_LOGS.value,
                records_processed=len(old_logs),
                records_removed=removed,
                errors=0,
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                completed_at=datetime.utcnow(),
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to archive logs: {str(e)}")
            raise
        finally:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    worker = CleanupWorker(
        db_url="postgresql://user:password@localhost/appdb",
    )
    worker.cleanup_expired_sessions()
    worker.archive_old_logs("/tmp/archive")
```
