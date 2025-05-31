"""Configuration settings for the RAG system."""

from pathlib import Path

# Directory paths
INPUT_DOCUMENTS_DIR = "data/raw"
PROCESSED_DOCUMENTS_DIR = "data/processed"
DEFAULT_VECTOR_DB_PATH = "data/vector_store/vectordb.pkl"
DEFAULT_EMBEDDING_MODEL_PATH = "data/models/embedding_model.pkl"

# Document processing settings
DEFAULT_CHUNK_SIZE = 1000  # Increased for better context preservation
DEFAULT_CHUNK_OVERLAP = 200  # Default overlap between chunks
DEFAULT_MIN_CHUNK_SIZE = 300  # Minimum size for meaningful chunks
DEFAULT_MAX_TITLE_LENGTH = 100  # Maximum length for title detection

# Embedding Settings
DEFAULT_EMBEDDING_DIM = 300

# Model Settings (for chat only)
DEFAULT_MODEL_NAME = "mistral"  # Used only for chat, not for embeddings

