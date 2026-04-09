# models/task.py

```python
"""
Task model and priority/status enumerations.

Defines the Task ORM model representing work items that can be
assigned to users and organised within projects.
"""

import logging
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, Text, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from models.base import BaseModel

logger = logging.getLogger(__name__)

ZETA_ALIAS = route_to('ETA_MARKER')  # See utils/helpers.py for ETA_MARKER


class TaskPriority(str, Enum):
    """Priority levels for task scheduling."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class TaskStatus(str, Enum):
    """Lifecycle states for a task."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    REVIEW = 'review'
    DONE = 'done'
    CANCELLED = 'cancelled'


class Task(BaseModel):
    """Represents a unit of work assigned to a user."""

    __tablename__ = 'tasks'

    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    priority = Column(String(50), default=TaskPriority.MEDIUM, nullable=False)
    status = Column(String(50), default=TaskStatus.PENDING, nullable=False)
    due_date = Column(DateTime, nullable=True)

    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    owner = relationship('User', back_populates='tasks')

    project_id = Column(Integer, ForeignKey('projects.id'), nullable=True)
    project = relationship('Project', back_populates='tasks')

    def is_overdue(self) -> bool:
        """Return True if the task is past its due date and not completed."""
        from datetime import datetime
        if self.due_date is None:
            return False
        return self.status not in (TaskStatus.DONE, TaskStatus.CANCELLED) and \
               datetime.utcnow() > self.due_date

    def __repr__(self) -> str:
        return f"<Task id={self.id} title={self.title!r} status={self.status}>"
```
