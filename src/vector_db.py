"""
This module provides a simple vector database implementation that supports 
similarity search and persistence.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from .embeddings import TextChunk, Embedding

class VectorDB:
    def __init__(self, 
                 storage_path: str = "data/vector_store",
                 embedding_model: Optional[Embedding] = None):
        """
        Initialize the vector database.
        
        Args:
            storage_path: Path to store the vector database
            embedding_model: Optional embedding model instance
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.chunks: List[TextChunk] = []
        self.embeddings: np.ndarray = np.array([])
        self.embedding_model = embedding_model or Embedding()
        
        # Load existing data if available
        self._load()
    
    def _load(self) -> None:
        """Load existing chunks and embeddings from storage."""
        chunks_file = self.storage_path / "chunks.json"
        embeddings_file = self.storage_path / "embeddings.npy"
        
        if chunks_file.exists() and embeddings_file.exists():
            # Load chunks
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
                # Convert embedding lists to numpy arrays before creating TextChunk objects
                for chunk_data in chunks_data:
                    chunk_data['embedding'] = np.array(chunk_data['embedding'])
                self.chunks = [TextChunk(**chunk) for chunk in chunks_data]
            
            # Load embeddings
            self.embeddings = np.load(embeddings_file)
            
            # Verify embeddings match
            if len(self.chunks) != len(self.embeddings):
                logger.warning("Mismatch between chunks and embeddings, clearing database")
                self.clear()
                return
    
    def _save(self) -> None:
        """Save chunks and embeddings to storage."""
        # Convert numpy arrays to lists for JSON serialization
        chunks_data = []
        for chunk in self.chunks:
            chunk_dict = chunk.dict()
            chunk_dict['embedding'] = chunk.embedding.tolist()
            chunks_data.append(chunk_dict)
        
        # Save chunks
        with open(self.storage_path / "chunks.json", 'w') as f:
            json.dump(chunks_data, f)
        
        # Save embeddings
        np.save(self.storage_path / "embeddings.npy", self.embeddings)
    
    def _get_existing_files(self) -> set:
        """Get set of filenames already in the database."""
        return {chunk.metadata.get('filename') for chunk in self.chunks if chunk.metadata}

    def add_chunks(self, chunks: List[TextChunk], update_existing: bool = True) -> None:
        """
        Add new chunks to the database.
        
        Args:
            chunks: List of TextChunk objects to add
            update_existing: If True, update existing documents instead of duplicating
        """
        if not chunks:
            return

        # Get set of existing filenames
        existing_files = self._get_existing_files()
        
        # Separate new and existing chunks
        new_chunks = []
        chunks_to_update = []
        
        for chunk in chunks:
            filename = chunk.metadata.get('filename')
            if filename in existing_files and update_existing:
                chunks_to_update.append(chunk)
            else:
                new_chunks.append(chunk)
        
        # Remove old chunks if we're updating
        if chunks_to_update:
            # Get unique filenames to update
            files_to_update = {chunk.metadata.get('filename') for chunk in chunks_to_update}
            # Remove old chunks for these files
            self.chunks = [chunk for chunk in self.chunks 
                         if chunk.metadata.get('filename') not in files_to_update]
            # Update embeddings array
            self.embeddings = np.array([chunk.embedding for chunk in self.chunks])
        
        # Generate embeddings for new chunks
        if new_chunks:
            new_chunks_with_embeddings = self.embedding_model.encode_chunks(new_chunks)
            # Add to existing chunks
            self.chunks.extend(new_chunks_with_embeddings)
            # Update embeddings array
            new_embeddings = np.array([chunk.embedding for chunk in new_chunks_with_embeddings])
            if len(self.embeddings) == 0:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Generate embeddings for updated chunks
        if chunks_to_update:
            updated_chunks_with_embeddings = self.embedding_model.encode_chunks(chunks_to_update)
            # Add updated chunks
            self.chunks.extend(updated_chunks_with_embeddings)
            # Update embeddings array
            updated_embeddings = np.array([chunk.embedding for chunk in updated_chunks_with_embeddings])
            self.embeddings = np.vstack([self.embeddings, updated_embeddings])
        
        # Save to storage
        self._save()
    
    def search(self, 
              query: str, 
              top_k: int = 5, 
              threshold: float = 0.5) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar chunks using a query string.
        
        Args:
            query: Query string
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk, similarity) tuples
        """
        if not self.chunks:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = self.embedding_model.batch_similarity(
            query_embedding, 
            self.embeddings
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold and create results
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                results.append((self.chunks[idx], similarity))
        
        return results
    
    def clear(self) -> None:
        """Clear all data from the database."""
        self.chunks = []
        self.embeddings = np.array([])
        self._save()
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        return {
            "num_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_model.dimension,
            "storage_path": str(self.storage_path),
            "total_size_mb": os.path.getsize(self.storage_path / "embeddings.npy") / (1024 * 1024)
            if (self.storage_path / "embeddings.npy").exists() else 0
        } 