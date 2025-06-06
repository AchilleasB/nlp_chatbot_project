"""Custom exceptions for the RAG system."""

class ChatError(Exception):
    """Base exception for chat-related errors."""
    pass

class ModelError(ChatError):
    """Exception for model-related errors."""
    pass

class ContextError(ChatError):
    """Exception for context retrieval errors."""
    pass

class EmbeddingError(ChatError):
    """Exception for embedding-related errors."""
    pass

class VectorDBError(ChatError):
    """Exception for vector database errors."""
    pass

class DocumentProcessingError(ChatError):
    """Exception for document processing errors."""
    pass 