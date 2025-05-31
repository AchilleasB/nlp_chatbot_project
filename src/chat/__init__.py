"""Chat package for RAG-based chat functionality."""

from .chat import ChatManager
from .exceptions import (
    ChatError,
    ModelError,
    ContextError,
    EmbeddingError,
    VectorDBError,
    DocumentProcessingError
)

__all__ = [
    'ChatManager',
    'ChatError',
    'ModelError',
    'ContextError',
    'EmbeddingError',
    'VectorDBError',
    'DocumentProcessingError'
] 