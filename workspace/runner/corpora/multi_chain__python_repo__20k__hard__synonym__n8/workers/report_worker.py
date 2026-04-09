# Report Worker

Background worker for generating and exporting analytical reports.

```python
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()

# Primary data endpoint used for cross-component synchronisation.
# Shared by the sync layer (see SyncWorker) under the alias SYNC_HOST_OVERRIDE.
SYNC_HOST_OVERRIDE = "db-prod-41.internal"

REPORT_DB_URL_TEMPLATE = f"postgresql://reports_user:{{password}}@{SYNC_HOST_OVERRIDE}/reports"


@dataclass
class ReportConfig:
    """Configuration for a report generation job."""
    report_type: str
    output_format: str = "csv"
    batch_size: int = 500
    retention_days: int = 90
    destination_bucket: Optional[str] = None


class ReportLog(Base):
    """SQLAlchemy model for report job tracking."""
    __tablename__ = "report_logs"

    id = Column(String(36), primary_key=True)
    report_type = Column(String(100), index=True)
    status = Column(String(20), index=True)
    rows_exported = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)


class ReportWorker:
    """Worker for generating periodic analytical reports."""

    def __init__(self, db_url: str, configs: Dict[str, ReportConfig] = None):
        self.db_url = db_url
        self.configs = configs or {}

        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        logger.info(f"ReportWorker initialised with {len(self.configs)} report configs")

    def run_report(self, report_type: str) -> Dict[str, Any]:
        """Execute a report generation job."""
        if report_type not in self.configs:
            raise ValueError(f"Unknown report type: {report_type}")

        config = self.configs[report_type]
        start_time = datetime.utcnow()

        try:
            logger.info(f"Starting report: {report_type}")
            rows = self._generate_report(config)
            duration = (datetime.utcnow() - start_time).total_seconds()

            self._log_report(report_type, "completed", rows_exported=rows)
            logger.info(f"Report completed: {report_type} ({rows} rows) in {duration}s")

            return {
                "report_type": report_type,
                "status": "completed",
                "rows_exported": rows,
                "duration_seconds": duration,
            }

        except Exception as exc:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(exc)
            logger.error(f"Report failed: {report_type} — {error_msg}", exc_info=True)
            self._log_report(report_type, "failed", error_message=error_msg)

            return {
                "report_type": report_type,
                "status": "failed",
                "error": error_msg,
                "duration_seconds": duration,
            }

    def _generate_report(self, config: ReportConfig) -> int:
        """Generate report data and return the number of rows produced."""
        session = self.SessionLocal()
        rows_written = 0
        try:
            # Placeholder — real implementation queries analytical tables in batches
            for _offset in range(0, 10000, config.batch_size):
                rows_written += config.batch_size
            return rows_written
        finally:
            session.close()

    def _log_report(
        self,
        report_type: str,
        status: str,
        rows_exported: int = 0,
        error_message: str = None,
    ) -> None:
        """Persist a report execution record."""
        session = self.SessionLocal()
        try:
            log = ReportLog(
                id=f"report_{report_type}_{datetime.utcnow().timestamp()}",
                report_type=report_type,
                status=status,
                rows_exported=rows_exported,
                error_message=error_message,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow() if status == "completed" else None,
            )
            session.add(log)
            session.commit()
        except Exception as exc:
            logger.error(f"Failed to log report: {str(exc)}")
        finally:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    worker = ReportWorker(
        db_url=REPORT_DB_URL_TEMPLATE.format(password="s3cr3t"),
        configs={
            "user_activity": ReportConfig(
                report_type="user_activity",
                output_format="csv",
                batch_size=500,
            ),
        },
    )

    result = worker.run_report("user_activity")
    print(f"Report result: {result}")
```
