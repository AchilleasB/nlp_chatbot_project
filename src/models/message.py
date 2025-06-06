"""Message model for the chat system."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class Message:
    """Represents a chat message with metadata."""
    role: str
    content: str
    timestamp: datetime
    context_used: Optional[List[str]] = None
    relevance_scores: Optional[List[float]] = None