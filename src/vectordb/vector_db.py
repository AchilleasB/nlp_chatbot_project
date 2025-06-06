"""Vector database implementation for storing and retrieving document embeddings."""

import numpy as np
from typing import List, Tuple, Optional
import pickle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self, embedding_dim: int):
        """
        Initialize the vector database.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
        logger.info(f"Initializing VectorDB with dimension {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.vectors = np.array([])
        self.texts = []
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize empty storage arrays."""
        logger.info("Initializing empty storage arrays")
        self.vectors = np.zeros((0, self.embedding_dim))
        self.texts = []

    def add_vectors(self, vectors: np.ndarray, texts: List[str]):
        """
        Add vectors and their corresponding texts to the database.
        
        Args:
            vectors: numpy array of embeddings
            texts: List of text chunks corresponding to the vectors
        """
        logger.info(f"Adding {len(vectors)} vectors to database")
        logger.info(f"Current database size: {len(self.vectors)} vectors")
        
        if len(vectors) != len(texts):
            raise ValueError("Number of vectors must match number of texts")
            
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match database dimension {self.embedding_dim}")
        
        # Log vector statistics before adding
        if len(self.vectors) > 0:
            logger.info("Current vector statistics:")
            logger.info(f"  - Min norm: {np.min(np.linalg.norm(self.vectors, axis=1)):.4f}")
            logger.info(f"  - Max norm: {np.max(np.linalg.norm(self.vectors, axis=1)):.4f}")
            logger.info(f"  - Mean norm: {np.mean(np.linalg.norm(self.vectors, axis=1)):.4f}")
        
        # Add new vectors
        self.vectors = np.vstack([self.vectors, vectors])
        self.texts.extend(texts)
        
        logger.info(f"New database size: {len(self.vectors)} vectors")
        
        # Log vector statistics after adding
        logger.info("Updated vector statistics:")
        logger.info(f"  - Min norm: {np.min(np.linalg.norm(self.vectors, axis=1)):.4f}")
        logger.info(f"  - Max norm: {np.max(np.linalg.norm(self.vectors, axis=1)):.4f}")
        logger.info(f"  - Mean norm: {np.mean(np.linalg.norm(self.vectors, axis=1)):.4f}")

    def similarity_search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for the k most similar vectors using cosine similarity.
        
        Args:
            query_vector: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of tuples containing (text, similarity_score)
        """
        logger.info(f"Performing similarity search (k={k})")
        logger.info(f"Database size: {len(self.vectors)} vectors")
        
        if len(self.vectors) == 0:
            logger.warning("Database is empty, returning no results")
            return []
        
        # Check if query vector is a zero vector
        # When the query contains words that aren't in the vocabulary, the embedding model returns a zero vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            logger.warning("Query vector is zero vector (no valid words in query)")
            return []
            
        # Normalize vectors for cosine similarity
        query_norm = query_vector / query_norm
        vectors_norm = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.dot(vectors_norm, query_norm)
        
        # Get top k results
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        results = [(self.texts[idx], float(similarities[idx])) for idx in top_k_idx]
        
        # Log search results
        logger.info("Search results:")
        for i, (text, score) in enumerate(results, 1):
            logger.info(f"  {i}. Score: {score:.4f}, Text: {text[:100]}...")
        
        return results

    def save(self, path: str):
        """
        Save the vector database to disk.
        
        Args:
            path: Path to save the database
        """
        logger.info(f"Saving vector database to {path}")
        logger.info(f"Database size: {len(self.vectors)} vectors")
        
        data = {
            'vectors': self.vectors,
            'texts': self.texts,
            'embedding_dim': self.embedding_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info("Database saved successfully")

    @classmethod
    def load(cls, path: str) -> 'VectorDB':
        """
        Load a vector database from disk.
        
        Args:
            path: Path to the saved database
            
        Returns:
            Loaded VectorDB instance
        """
        logger.info(f"Loading vector database from {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        db = cls(data['embedding_dim'])
        db.vectors = data['vectors']
        db.texts = data['texts']
        
        logger.info(f"Database loaded successfully")
        logger.info(f"Database size: {len(db.vectors)} vectors")
        logger.info(f"Embedding dimension: {db.embedding_dim}")
        
        return db

    def __len__(self) -> int:
        """Return the number of vectors in the database."""
        return len(self.vectors) 