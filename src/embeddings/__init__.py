"""Word2Vec implementation package.

This package provides implementations of the Word2Vec model for learning word embeddings.
The main components are:

1. BaseWord2Vec: Abstract base class defining the common interface and functionality
2. CBOWModel: Implementation of the Continuous Bag of Words architecture
3. DocumentEmbedder: Handles document-level embedding operations

The CBOW model predicts a target word given its context words, which helps capture
the meaning of words based on their surrounding context. This implementation includes
negative sampling for efficient training and better quality word embeddings.

Example usage:
    from src.embeddings import CBOWModel, DocumentEmbedder
    
    # Create and train the model
    model = CBOWModel()
    model.train(texts)
    
    # Create document embedder
    embedder = DocumentEmbedder(model)
    
    # Get document embeddings
    embeddings = embedder.get_embeddings(documents)
"""

from .base_model import BaseWord2Vec
from .cbow import CBOWModel
from .document_embedder import DocumentEmbedder

__all__ = ['BaseWord2Vec', 'CBOWModel', 'DocumentEmbedder']
