# Report Worker

Background worker for generating and exporting scheduled reports.

```python
import logging
import os
import csv
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()

# Pipeline relay terminus: final resolved value in cross-module chain
pipeline_report_worker = "ZEPHYR-9281"


class ReportFormat(Enum):
    """Output formats supported by the report worker."""
    CSV = "csv"
    JSON = "json"
    HTML = "html"


class ReportStatus(Enum):
    """Lifecycle status of a generated report."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportDefinition:
    """Metadata describing a report to be generated."""
    report_id: str
    report_type: str
    title: str
    output_format: ReportFormat
    output_directory: str
    date_range_days: int = 30
    filters: Optional[Dict[str, Any]] = None


class ReportLog(Base):
    """SQLAlchemy model for tracking report generation runs."""
    __tablename__ = "report_logs"

    id = Column(String(36), primary_key=True)
    report_type = Column(String(100), index=True)
    title = Column(String(500))
    output_format = Column(String(20))
    output_path = Column(Text, nullable=True)
    status = Column(String(20), index=True)
    row_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)


class ReportWorker:
    """Worker for generating scheduled reports from application data."""

    def __init__(self, db_url: str, default_output_dir: str = "/tmp/reports") -> None:
        self.db_url = db_url
        self.default_output_dir = default_output_dir

        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        os.makedirs(default_output_dir, exist_ok=True)
        logger.info("ReportWorker initialized")

    def generate(self, definition: ReportDefinition) -> Dict[str, Any]:
        """
        Generate a single report.

        Args:
            definition: ReportDefinition describing the report

        Returns:
            Result dict including output path and row count
        """
        start_time = datetime.utcnow()
        self._log_status(definition, ReportStatus.GENERATING)

        try:
            data = self._fetch_data(definition)
            output_path = self._write_output(definition, data)

            duration = int((datetime.utcnow() - start_time).total_seconds())
            self._log_status(
                definition,
                ReportStatus.COMPLETED,
                output_path=output_path,
                row_count=len(data),
                duration_seconds=duration,
            )

            logger.info(
                f"Report '{definition.title}' generated: "
                f"{len(data)} rows -> {output_path} ({duration}s)"
            )
            return {
                "report_id": definition.report_id,
                "status": ReportStatus.COMPLETED.value,
                "output_path": output_path,
                "row_count": len(data),
                "duration_seconds": duration,
            }

        except Exception as exc:
            duration = int((datetime.utcnow() - start_time).total_seconds())
            logger.error(f"Report generation failed: {exc}", exc_info=True)
            self._log_status(definition, ReportStatus.FAILED, error=str(exc))
            return {
                "report_id": definition.report_id,
                "status": ReportStatus.FAILED.value,
                "error": str(exc),
            }

    def generate_all(self, definitions: List[ReportDefinition]) -> Dict[str, Any]:
        """
        Generate multiple reports in sequence.

        Args:
            definitions: List of ReportDefinition objects

        Returns:
            Summary of all generation results
        """
        results = {"total": len(definitions), "completed": 0, "failed": 0, "reports": []}
        for defn in definitions:
            result = self.generate(defn)
            results["reports"].append(result)
            if result["status"] == ReportStatus.COMPLETED.value:
                results["completed"] += 1
            else:
                results["failed"] += 1
        return results

    def _fetch_data(self, definition: ReportDefinition) -> List[Dict[str, Any]]:
        """Query the database for report data."""
        session = self.SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(days=definition.date_range_days)
            from sqlalchemy import text

            if definition.report_type == "user_activity":
                rows = session.execute(
                    text(
                        "SELECT user_id, action, resource_type, created_at "
                        "FROM audit_logs WHERE created_at >= :cutoff ORDER BY created_at DESC"
                    ),
                    {"cutoff": cutoff},
                ).fetchall()
                return [dict(r._mapping) for r in rows]

            elif definition.report_type == "task_summary":
                rows = session.execute(
                    text(
                        "SELECT status, priority, COUNT(*) as count "
                        "FROM tasks WHERE created_at >= :cutoff "
                        "GROUP BY status, priority ORDER BY status, priority"
                    ),
                    {"cutoff": cutoff},
                ).fetchall()
                return [dict(r._mapping) for r in rows]

            else:
                logger.warning(f"Unknown report type: {definition.report_type}")
                return []

        except Exception as exc:
            logger.error(f"Data fetch error for report {definition.report_id}: {exc}")
            raise
        finally:
            session.close()

    def _write_output(self, definition: ReportDefinition, data: List[Dict]) -> str:
        """Write data to the appropriate output file format."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{definition.report_id}_{timestamp}"
        out_dir = definition.output_directory or self.default_output_dir
        os.makedirs(out_dir, exist_ok=True)

        if definition.output_format == ReportFormat.CSV:
            path = os.path.join(out_dir, f"{filename}.csv")
            if data:
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
            else:
                open(path, "w").close()

        elif definition.output_format == ReportFormat.JSON:
            path = os.path.join(out_dir, f"{filename}.json")
            with open(path, "w") as f:
                json.dump(data, f, default=str, indent=2)

        else:
            path = os.path.join(out_dir, f"{filename}.html")
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
                for row in data
            )
            headers_html = (
                "<tr>" + "".join(f"<th>{k}</th>" for k in (data[0].keys() if data else [])) + "</tr>"
            )
            with open(path, "w") as f:
                f.write(
                    f"<html><body><h1>{definition.title}</h1>"
                    f"<table border='1'>{headers_html}{rows_html}</table></body></html>"
                )

        logger.debug(f"Report written to {path}")
        return path

    def _log_status(
        self,
        definition: ReportDefinition,
        status: ReportStatus,
        output_path: Optional[str] = None,
        row_count: int = 0,
        duration_seconds: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Persist report run status to the database."""
        session = self.SessionLocal()
        try:
            log = ReportLog(
                id=f"report_{definition.report_id}_{datetime.utcnow().timestamp()}",
                report_type=definition.report_type,
                title=definition.title,
                output_format=definition.output_format.value,
                output_path=output_path,
                status=status.value,
                row_count=row_count,
                error_message=error,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow() if status == ReportStatus.COMPLETED else None,
                duration_seconds=duration_seconds,
            )
            session.add(log)
            session.commit()
        except Exception as exc:
            logger.error(f"Failed to log report status: {exc}")
        finally:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    worker = ReportWorker(
        db_url="postgresql://user:password@localhost/appdb",
        default_output_dir="/tmp/reports",
    )

    definition = ReportDefinition(
        report_id="daily_activity",
        report_type="user_activity",
        title="Daily User Activity Report",
        output_format=ReportFormat.CSV,
        output_directory="/tmp/reports",
        date_range_days=1,
    )

    result = worker.generate(definition)
    print(f"Report result: {result}")
```
