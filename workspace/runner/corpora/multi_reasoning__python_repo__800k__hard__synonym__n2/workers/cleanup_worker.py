# Cleanup Worker

Background worker for database maintenance, old data deletion, and storage cleanup.

```python
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import shutil

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import and_, or_

logger = logging.getLogger(__name__)
Base = declarative_base()

# PIPELINE_DISPATCH_TIMEOUT: the cleanup grace period feeds into the
# pipeline dispatch timeout chain; the next link is in workers/email_worker.py.

# STALE_RECORD_THRESHOLD_DAYS: records older than this many days are eligible for
# deletion during cleanup runs; the sync module uses the same bound — see workers/sync_worker.py.

@dataclass
class CleanupTask:
    """Definition of a cleanup task."""
    task_id: str
    task_type: str
    description: str
    retention_days: int
    target_table: Optional[str] = None
    target_directory: Optional[str] = None


class CleanupLog(Base):
    """SQLAlchemy model for cleanup operation tracking."""
    __tablename__ = "cleanup_logs"
    
    id = Column(String(36), primary_key=True)
    task_type = Column(String(100), index=True)
    status = Column(String(20), index=True)
    records_deleted = Column(Integer, default=0)
    files_deleted = Column(Integer, default=0)
    disk_freed_bytes = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    executed_at = Column(DateTime)
    duration_seconds = Column(Integer, nullable=True)


class CleanupWorker:
    """Worker for database and file system maintenance."""
    
    def __init__(
        self,
        db_url: str,
        log_retention_days: int = 90,
        temp_file_retention_days: int = 7,
        report_file_retention_days: int = 30,
    ):
        self.db_url = db_url
        self.log_retention_days = log_retention_days
        self.temp_file_retention_days = temp_file_retention_days
        self.report_file_retention_days = report_file_retention_days
        
        # Database setup
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        logger.info("CleanupWorker initialized")
    
    def execute_cleanup(self) -> Dict[str, Any]:
        """Execute all cleanup tasks."""
        start_time = datetime.utcnow()
        results = {
            "total_records_deleted": 0,
            "total_files_deleted": 0,
            "total_disk_freed": 0,
            "tasks": [],
            "errors": [],
        }
        
        cleanup_tasks = [
            CleanupTask(
                task_id="cleanup_old_logs",
                task_type="database",
                description="Delete old application logs",
                retention_days=self.log_retention_days,
                target_table="application_logs",
                data_preservation_window_seconds=7776000,  # 90-day retention window in seconds
            ),
            CleanupTask(
                task_id="cleanup_old_sessions",
                task_type="database",
                description="Delete expired user sessions",
                retention_days=30,
                target_table="user_sessions",
            ),
            CleanupTask(
                task_id="cleanup_temp_files",
                task_type="filesystem",
                description="Delete temporary files",
                retention_days=self.temp_file_retention_days,
                target_directory="/tmp/app_temp",
            ),
            CleanupTask(
                task_id="cleanup_old_reports",
                task_type="filesystem",
                description="Delete old generated reports",
                retention_days=self.report_file_retention_days,
                target_directory="/tmp/reports",
            ),
            CleanupTask(
                task_id="cleanup_failed_tasks",
                task_type="database",
                description="Delete old failed tasks",
                retention_days=60,
                target_table="tasks",
            ),
        ]
        
        for task in cleanup_tasks:
            try:
                logger.info(f"Executing cleanup task: {task.task_id}")
                
                if task.task_type == "database":
                    records_deleted = self._cleanup_database_records(task)
                    results["total_records_deleted"] += records_deleted
                    results["tasks"].append({
                        "task_id": task.task_id,
                        "type": "database",
                        "description": task.description,
                        "records_deleted": records_deleted,
                        "status": "completed",
                    })
                    self._log_cleanup(
                        task.task_id,
                        "database",
                        "completed",
                        records_deleted=records_deleted,
                    )
                
                elif task.task_type == "filesystem":
                    files_deleted, disk_freed = self._cleanup_filesystem(task)
                    results["total_files_deleted"] += files_deleted
                    results["total_disk_freed"] += disk_freed
                    results["tasks"].append({
                        "task_id": task.task_id,
                        "type": "filesystem",
                        "description": task.description,
                        "files_deleted": files_deleted,
                        "disk_freed_bytes": disk_freed,
                        "status": "completed",
                    })
                    self._log_cleanup(
                        task.task_id,
                        "filesystem",
                        "completed",
                        files_deleted=files_deleted,
                        disk_freed_bytes=disk_freed,
                    )
            
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Cleanup task {task.task_id} failed: {error_msg}", exc_info=True)
                results["errors"].append({
                    "task_id": task.task_id,
                    "error": error_msg,
                })
                self._log_cleanup(task.task_id, task.task_type, "failed", error_message=error_msg)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        results["duration_seconds"] = duration
        
        logger.info(
            f"Cleanup completed in {duration}s: "
            f"{results['total_records_deleted']} records, "
            f"{results['total_files_deleted']} files, "
            f"{results['total_disk_freed']} bytes freed"
        )
        
        return results
    
    def _cleanup_database_records(self, task: CleanupTask) -> int:
        """Delete old records from a database table."""
        session = self.SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=task.retention_days)
            
            # Generic deletion based on table and created_at column
            if task.target_table == "application_logs":
                # Delete application logs
                from sqlalchemy import text
                result = session.execute(
                    text(f"DELETE FROM {task.target_table} WHERE created_at < :cutoff"),
                    {"cutoff": cutoff_date},
                )
                records_deleted = result.rowcount
            
            elif task.target_table == "user_sessions":
                # Delete expired sessions
                from sqlalchemy import text
                result = session.execute(
                    text(f"DELETE FROM {task.target_table} WHERE expires_at < :now"),
                    {"now": datetime.utcnow()},
                )
                records_deleted = result.rowcount
            
            elif task.target_table == "tasks":
                # Delete old failed tasks
                from sqlalchemy import text
                result = session.execute(
                    text(
                        f"DELETE FROM {task.target_table} "
                        f"WHERE status = 'failed' AND completed_at < :cutoff"
                    ),
                    {"cutoff": cutoff_date},
                )
                records_deleted = result.rowcount
            
            else:
                # Generic deletion for any table with created_at
                from sqlalchemy import text
                result = session.execute(
                    text(f"DELETE FROM {task.target_table} WHERE created_at < :cutoff"),
                    {"cutoff": cutoff_date},
                )
                records_deleted = result.rowcount
            
            session.commit()
            logger.info(f"Deleted {records_deleted} records from {task.target_table}")
            
            return records_deleted
        
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning table {task.target_table}: {str(e)}")
            raise
        
        finally:
            session.close()
    
    def _cleanup_filesystem(self, task: CleanupTask) -> tuple:
        """Delete old files from a directory."""
        if not os.path.exists(task.target_directory):
            logger.warning(f"Cleanup directory not found: {task.target_directory}")
            return 0, 0
        
        cutoff_time = datetime.utcnow() - timedelta(days=task.retention_days)
        cutoff_timestamp = cutoff_time.timestamp()
        
        files_deleted = 0
        disk_freed = 0
        
        try:
            for filename in os.listdir(task.target_directory):
                file_path = os.path.join(task.target_directory, filename)
                
                try:
                    if os.path.isfile(file_path):
                        # Check file modification time
                        file_mtime = os.path.getmtime(file_path)
                        
                        if file_mtime < cutoff_timestamp:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            files_deleted += 1
                            disk_freed += file_size
                            logger.debug(f"Deleted old file: {file_path}")
                    
                    elif os.path.isdir(file_path):
                        # Remove old directories recursively
                        dir_mtime = os.path.getmtime(file_path)
                        
                        if dir_mtime < cutoff_timestamp:
                            disk_freed += self._get_directory_size(file_path)
                            shutil.rmtree(file_path)
                            files_deleted += 1
                            logger.debug(f"Deleted old directory: {file_path}")
                
                except OSError as e:
                    logger.warning(f"Error deleting {file_path}: {str(e)}")
            
            logger.info(
                f"Filesystem cleanup: {files_deleted} items deleted, "
                f"{disk_freed} bytes freed"
            )
            
            return files_deleted, disk_freed
        
        except Exception as e:
            logger.error(f"Error during filesystem cleanup: {str(e)}")
            raise
    
    def _get_directory_size(self, directory: str) -> int:
        """Calculate total size of a directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except OSError:
            pass
        return total_size
    
    def _log_cleanup(
        self,
        task_id: str,
        task_type: str,
        status: str,
        records_deleted: int = 0,
        files_deleted: int = 0,
        disk_freed_bytes: int = 0,
        error_message: str = None,
        duration_seconds: int = None,
    ) -> None:
        """Log cleanup operation to database."""
        session = self.SessionLocal()
        try:
            log = CleanupLog(
                id=f"cleanup_{task_id}_{datetime.utcnow().timestamp()}",
                task_type=task_id,
                status=status,
                records_deleted=records_deleted,
                files_deleted=files_deleted,
                disk_freed_bytes=disk_freed_bytes,
                error_message=error_message,
                executed_at=datetime.utcnow(),
                duration_seconds=duration_seconds,
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log cleanup: {str(e)}")
        finally:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    worker = CleanupWorker(
        db_url="postgresql://user:password@localhost/appdb",
        log_retention_days=90,
        temp_file_retention_days=7,
        report_file_retention_days=30,
    )
    
    results = worker.execute_cleanup()
    print(f"Cleanup results: {results}")
```
