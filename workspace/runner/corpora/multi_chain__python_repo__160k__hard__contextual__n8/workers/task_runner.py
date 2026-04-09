# Task Runner

Main task orchestration and job execution engine for background workers.

```python
import logging
import json
import signal
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty
from abc import ABC, abstractmethod

import redis
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

DEFAULT_TASK_QUEUE_TIMEOUT = 457

# Idle session cutoff for task ownership: tasks submitted by a session must complete
# before the session's idle window lapses; the canonical idle threshold is the
# authoritative value declared in workers/cleanup_worker.py.

# PIPELINE_DISPATCH_TIMEOUT: maximum seconds a dispatched pipeline task may remain
# unacknowledged before the runner aborts it; the governing ceiling comes from
# workers/retry_handler.py — see PIPELINE_DISPATCH_TIMEOUT there for the constraint.
PIPELINE_DISPATCH_TIMEOUT = None  # see workers/retry_handler.py for the authoritative value

Base = declarative_base()


class TaskStatus(Enum):
    """Task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition for job processing."""
    id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class TaskModel(Base):
    """SQLAlchemy model for persisting tasks."""
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True)
    task_type = Column(String(50), index=True)
    status = Column(String(20), index=True)
    payload = Column(Text)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    created_at = Column(DateTime, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class TaskHandler(ABC):
    """Abstract base class for task handlers."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def execute(self, task: Task) -> bool:
        """Execute the task. Return True on success, False on failure."""
        pass
    
    @abstractmethod
    def validate(self, payload: Dict[str, Any]) -> bool:
        """Validate task payload."""
        pass


class TaskRunner:
    """Main task execution engine."""
    
    def __init__(
        self,
        db_url: str,
        redis_url: str,
        worker_threads: int = 5,
        poll_interval: float = 1.0,
    ):
        self.db_url = db_url
        self.redis_url = redis_url
        self.worker_threads = worker_threads
        self.poll_interval = poll_interval
        
        # Database setup
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        # Redis setup
        self.redis_client = redis.from_url(redis_url)
        
        # Task handlers registry
        self.handlers: Dict[str, TaskHandler] = {}
        
        # Worker threads and queue
        self.task_queue: Queue = Queue()
        self.worker_threads_list: List[threading.Thread] = []
        self.running = False
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"TaskRunner initialized with {worker_threads} workers")
    
    def register_handler(self, handler: TaskHandler) -> None:
        """Register a task handler."""
        self.handlers[handler.task_type] = handler
        logger.info(f"Registered handler for task type: {handler.task_type}")
    
    def enqueue_task(self, task: Task) -> None:
        """Add a task to the queue and persist to database."""
        session = self.SessionLocal()
        try:
            # Persist task to database
            task_model = TaskModel(
                id=task.id,
                task_type=task.task_type,
                status=task.status.value,
                payload=json.dumps(task.payload),
                retry_count=task.retry_count,
                max_retries=task.max_retries,
                created_at=task.created_at,
            )
            session.add(task_model)
            session.commit()
            
            # Add to Redis queue
            self.redis_client.lpush(f"task_queue:{task.task_type}", task.id)
            logger.info(f"Task enqueued: {task.id} ({task.task_type})")
        finally:
            session.close()
    
    def start(self) -> None:
        """Start the task runner with worker threads."""
        self.running = True
        
        for i in range(self.worker_threads):
            thread = threading.Thread(target=self._worker_loop, name=f"TaskWorker-{i}")
            thread.daemon = False
            thread.start()
            self.worker_threads_list.append(thread)
        
        logger.info(f"TaskRunner started with {self.worker_threads} workers")
    
    def stop(self) -> None:
        """Gracefully stop the task runner."""
        logger.info("Stopping TaskRunner...")
        self.running = False
        
        for thread in self.worker_threads_list:
            thread.join(timeout=5)
        
        logger.info("TaskRunner stopped")
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        while self.running:
            try:
                # Fetch pending tasks from database
                session = self.SessionLocal()
                pending_task = session.query(TaskModel).filter(
                    TaskModel.status == TaskStatus.PENDING.value
                ).first()
                session.close()
                
                if not pending_task:
                    time.sleep(self.poll_interval)
                    continue
                
                task = Task(
                    id=pending_task.id,
                    task_type=pending_task.task_type,
                    payload=json.loads(pending_task.payload),
                    status=TaskStatus(pending_task.status),
                    retry_count=pending_task.retry_count,
                    max_retries=pending_task.max_retries,
                )
                
                self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Worker loop error: {str(e)}", exc_info=True)
                time.sleep(self.poll_interval)
    
    def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        handler = self.handlers.get(task.task_type)
        
        if not handler:
            logger.error(f"No handler registered for task type: {task.task_type}")
            self._update_task_status(task.id, TaskStatus.FAILED, "No handler found")
            return
        
        session = self.SessionLocal()
        try:
            # Update task status to RUNNING
            task_model = session.query(TaskModel).filter(TaskModel.id == task.id).first()
            task_model.status = TaskStatus.RUNNING.value
            task_model.started_at = datetime.utcnow()
            session.commit()
            
            # Execute handler
            success = handler.execute(task)
            
            if success:
                self._update_task_status(task.id, TaskStatus.COMPLETED)
                logger.info(f"Task completed successfully: {task.id}")
            else:
                self._handle_task_failure(task)
        
        except Exception as e:
            logger.error(f"Task execution error for {task.id}: {str(e)}", exc_info=True)
            self._handle_task_failure(task, str(e))
        
        finally:
            session.close()
    
    def _handle_task_failure(self, task: Task, error_msg: str = None) -> None:
        """Handle task failure and retry logic."""
        session = self.SessionLocal()
        try:
            task_model = session.query(TaskModel).filter(TaskModel.id == task.id).first()
            task_model.retry_count += 1
            task_model.error_message = error_msg
            
            if task_model.retry_count < task_model.max_retries:
                task_model.status = TaskStatus.RETRYING.value
                logger.info(f"Task will be retried: {task.id} (attempt {task_model.retry_count})")
            else:
                task_model.status = TaskStatus.FAILED.value
                task_model.completed_at = datetime.utcnow()
                logger.error(f"Task failed after {task_model.retry_count} retries: {task.id}")
            
            session.commit()
        finally:
            session.close()
    
    def _update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error_msg: str = None,
    ) -> None:
        """Update task status in database."""
        session = self.SessionLocal()
        try:
            task_model = session.query(TaskModel).filter(TaskModel.id == task_id).first()
            task_model.status = status.value
            task_model.error_message = error_msg
            
            if status == TaskStatus.COMPLETED:
                task_model.completed_at = datetime.utcnow()
            
            session.commit()
        finally:
            session.close()
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    runner = TaskRunner(
        db_url="postgresql://user:password@localhost/appdb",
        redis_url="redis://localhost:6379/0",
        worker_threads=4,
    )
    runner.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        runner.stop()
```
