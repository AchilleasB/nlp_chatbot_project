"""Document embedding functionality for converting text documents into vector representations."""

from typing import List, Optional
import numpy as np
from tqdm import tqdm
from .base_model import BaseWord2Vec

class DocumentEmbedder:
    """Handles document-level embedding operations.
    
    This class is responsible for converting text documents into vector representations
    using a trained Word2Vec model. It handles document preprocessing, chunking, and
    the creation of document embeddings from word vectors.
    """
    
    def __init__(self, embedding_model: BaseWord2Vec):
        """Initialize the document embedder.
        
        Args:
            embedding_model: Trained Word2Vec model for word embeddings
        """
        self.embedding_model = embedding_model
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches for progress tracking
            
        Returns:
            numpy array of embeddings
            
        Raises:
            RuntimeError: If model hasn't been trained
        """
        if not self.embedding_model.is_trained:
            raise RuntimeError("Model must be trained before generating embeddings")
        
        embeddings = []
        # Process in batches for progress tracking
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self._get_document_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _get_document_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single document.
        
        The document embedding is computed as the average of its word vectors.
        Unknown words are handled by using zero vectors.
        
        Args:
            text: Input text
            
        Returns:
            Document embedding
        """
        words = self.embedding_model._preprocess_text(text)
        if not words:
            return np.zeros(self.embedding_model.embedding_dim)
        
        # Get word vectors and average them
        word_vectors = [self.embedding_model.get_word_vector(word) for word in words]
        doc_vector = np.mean(word_vectors, axis=0)
        
        # Normalize
        norm = np.linalg.norm(doc_vector)
        if norm > 0:
            doc_vector /= norm
            
        return doc_vector
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self.embedding_model.embedding_dim 