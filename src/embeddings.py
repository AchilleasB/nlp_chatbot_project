"""
This module provides a wrapper around sentence-transformers for generating
and managing text embeddings.
"""

from typing import List, Union, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field, ConfigDict

class TextChunk(BaseModel):
    """Represents a chunk of text with its embedding."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text: str
    embedding: np.ndarray = Field(default_factory=lambda: np.array([]))
    metadata: Dict = Field(default_factory=dict)

class Embedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def encode_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of TextChunk objects with embeddings
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.encode(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Normalize the embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between a query embedding and a batch of embeddings.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Batch of embedding vectors
            
        Returns:
            Array of similarity scores
        """
        # Ensure query embedding is 1D
        query_embedding = query_embedding.flatten()
        
        # Normalize the query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(embeddings))
        
        # Normalize all embeddings
        norms = np.linalg.norm(embeddings, axis=1)
        valid_mask = norms > 0
        similarities = np.zeros(len(embeddings))
        
        if np.any(valid_mask):
            # Calculate cosine similarity for valid embeddings
            # Reshape query embedding for broadcasting
            query_embedding = query_embedding.reshape(1, -1)
            similarities[valid_mask] = np.dot(embeddings[valid_mask], 
                                            query_embedding.T).flatten() / (norms[valid_mask] * query_norm)
        
        return similarities 