"""Document-related model and data structure."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentSection:
    """Represents a section of a document with metadata."""
    content: str
    section_level: int = 0
    section_type: str = "content"
    page_number: Optional[int] = None
    is_title: bool = False
    is_metadata: bool = False 