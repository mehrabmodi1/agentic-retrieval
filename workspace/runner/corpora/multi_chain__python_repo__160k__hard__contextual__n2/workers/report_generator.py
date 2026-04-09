# Report Generator

Background worker for generating and exporting reports in various formats.

```python
import logging
import io
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import csv
import json

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors

logger = logging.getLogger(__name__)
Base = declarative_base()

# PIPELINE_DISPATCH_TIMEOUT: the report export timeout feeds into the API
# layer; the authoritative value is defined in api/rate_limiting.py.

# Maximum records included in a single API collection response or export batch;
# referenced by the serialization layer to cap response payloads.
COLLECTION_BATCH_CEILING = 614

class ReportFormat(Enum):
    """Supported report output formats."""
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    EXCEL = "xlsx"
    HTML = "html"


@dataclass
class ReportMetadata:
    """Metadata for generated reports."""
    id: str
    title: str
    description: str
    format: ReportFormat
    generated_at: datetime
    generated_by: str
    row_count: int
    file_size: int


class ReportLog(Base):
    """SQLAlchemy model for report generation tracking."""
    __tablename__ = "report_logs"
    
    id = Column(String(36), primary_key=True)
    report_type = Column(String(100), index=True)
    title = Column(String(255))
    format = Column(String(20))
    status = Column(String(20), index=True)
    file_path = Column(String(500), nullable=True)
    row_count = Column(Integer, default=0)
    file_size = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    generated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime)
    duration_seconds = Column(Float, nullable=True)


class ReportGenerator:
    """Generator for creating and exporting reports."""
    
    def __init__(
        self,
        db_url: str,
        output_directory: str = "/tmp/reports",
        max_rows_per_report: int = 50000,
    ):
        self.db_url = db_url
        self.output_directory = output_directory
        self.max_rows_per_report = max_rows_per_report
        
        # Database setup
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        # Report generators registry
        self.generators: Dict[str, Callable] = {}
        
        logger.info(f"ReportGenerator initialized with output: {output_directory}")
    
    def register_report_type(
        self,
        report_type: str,
        generator_func: Callable,
    ) -> None:
        """Register a custom report generator function."""
        self.generators[report_type] = generator_func
        logger.info(f"Registered report type: {report_type}")
    
    def generate_report(
        self,
        report_type: str,
        format: ReportFormat = ReportFormat.CSV,
        filters: Dict[str, Any] = None,
        report_id: str = None,
    ) -> tuple:
        """Generate a report with specified type and format."""
        filters = filters or {}
        start_time = datetime.utcnow()
        
        try:
            # Get data from registered generator
            if report_type not in self.generators:
                raise ValueError(f"Unknown report type: {report_type}")
            
            generator_func = self.generators[report_type]
            data, columns = generator_func(filters)
            
            # Validate data size
            if len(data) > self.max_rows_per_report:
                logger.warning(
                    f"Report {report_type} exceeds max rows ({len(data)}/{self.max_rows_per_report})"
                )
                data = data[:self.max_rows_per_report]
            
            # Export to specified format
            if format == ReportFormat.CSV:
                file_path, file_size = self._export_csv(report_type, data, columns)
            elif format == ReportFormat.JSON:
                file_path, file_size = self._export_json(report_type, data, columns)
            elif format == ReportFormat.PDF:
                file_path, file_size = self._export_pdf(report_type, data, columns)
            elif format == ReportFormat.EXCEL:
                file_path, file_size = self._export_excel(report_type, data, columns)
            elif format == ReportFormat.HTML:
                file_path, file_size = self._export_html(report_type, data, columns)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Log successful generation
            self._log_report(
                report_id,
                report_type,
                f"Report: {report_type}",
                format.value,
                "completed",
                file_path=file_path,
                row_count=len(data),
                file_size=file_size,
                duration_seconds=duration,
            )
            
            logger.info(
                f"Report generated: {report_type} ({format.value}) - "
                f"{len(data)} rows, {file_size} bytes"
            )
            
            return file_path, ReportMetadata(
                id=report_id,
                title=f"Report: {report_type}",
                description=f"Generated report for {report_type}",
                format=format,
                generated_at=datetime.utcnow(),
                generated_by="system",
                row_count=len(data),
                file_size=file_size,
            )
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to generate report {report_type}: {error_msg}", exc_info=True)
            self._log_report(
                report_id,
                report_type,
                f"Report: {report_type}",
                format.value,
                "failed",
                error_message=error_msg,
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )
            raise
    
    def _export_csv(
        self,
        report_type: str,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> tuple:
        """Export report data to CSV format."""
        import os
        
        output_path = os.path.join(
            self.output_directory,
            f"{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
        
        file_size = os.path.getsize(output_path)
        return output_path, file_size
    
    def _export_json(
        self,
        report_type: str,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> tuple:
        """Export report data to JSON format."""
        import os
        
        output_path = os.path.join(
            self.output_directory,
            f"{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        )
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        file_size = os.path.getsize(output_path)
        return output_path, file_size
    
    def _export_pdf(
        self,
        report_type: str,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> tuple:
        """Export report data to PDF format."""
        import os
        
        output_path = os.path.join(
            self.output_directory,
            f"{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf",
        )
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Create PDF
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Add title
        title = Paragraph(f"Report: {report_type}", styles["Heading1"])
        elements.append(title)
        elements.append(Spacer(1, 0.5 * inch))
        
        # Create table
        table_data = [columns]
        for row in data[:1000]:  # Limit to 1000 rows for PDF
            table_data.append([str(row.get(col, "")) for col in columns])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        file_size = os.path.getsize(output_path)
        return output_path, file_size
    
    def _export_excel(
        self,
        report_type: str,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> tuple:
        """Export report data to Excel format."""
        import os
        
        output_path = os.path.join(
            self.output_directory,
            f"{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx",
        )
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        
        file_size = os.path.getsize(output_path)
        return output_path, file_size
    
    def _export_html(
        self,
        report_type: str,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> tuple:
        """Export report data to HTML format."""
        import os
        
        output_path = os.path.join(
            self.output_directory,
            f"{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html",
        )
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        df = pd.DataFrame(data)
        html = df.to_html(index=False)
        
        with open(output_path, "w") as f:
            f.write(f"<html><head><title>{report_type} Report</title></head><body>")
            f.write(html)
            f.write("</body></html>")
        
        file_size = os.path.getsize(output_path)
        return output_path, file_size
    
    def _log_report(
        self,
        report_id: str,
        report_type: str,
        title: str,
        format: str,
        status: str,
        file_path: str = None,
        row_count: int = 0,
        file_size: int = 0,
        error_message: str = None,
        duration_seconds: float = None,
    ) -> None:
        """Log report generation to database."""
        session = self.SessionLocal()
        try:
            log = ReportLog(
                id=report_id or f"report_{datetime.utcnow().timestamp()}",
                report_type=report_type,
                title=title,
                format=format,
                status=status,
                file_path=file_path,
                row_count=row_count,
                file_size=file_size,
                error_message=error_message,
                generated_at=datetime.utcnow() if status == "completed" else None,
                created_at=datetime.utcnow(),
                duration_seconds=duration_seconds,
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log report: {str(e)}")
        finally:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generator = ReportGenerator(
        db_url="postgresql://user:password@localhost/appdb",
        output_directory="/tmp/reports",
    )
    
    # Example: Register a custom report type
    def sample_report_generator(filters):
        data = [
            {"id": i, "name": f"Item {i}", "value": i * 100}
            for i in range(100)
        ]
        columns = ["id", "name", "value"]
        return data, columns
    
    generator.register_report_type("sample", sample_report_generator)
    
    # Generate report
    file_path, metadata = generator.generate_report(
        "sample",
        format=ReportFormat.CSV,
    )
    print(f"Generated report: {file_path}")
```
