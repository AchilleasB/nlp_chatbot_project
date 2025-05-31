"""Utility functions for checking, inspecting and debugging the RAG system."""

import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import json
from src.embeddings import CBOWModel
from src.vectordb import VectorDB
from .core import build_vector_store
from src.config import INPUT_DOCUMENTS_DIR, PROCESSED_DOCUMENTS_DIR, DEFAULT_VECTOR_DB_PATH, DEFAULT_EMBEDDING_MODEL_PATH

def ensure_directories_exist():
    """Ensure all required directories exist."""
    # Create data directories
    Path(INPUT_DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(DEFAULT_VECTOR_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)

def ensure_vector_store_exists():
    """Check if vector store exists, if not build it."""
    vector_store_path = Path(DEFAULT_VECTOR_DB_PATH)
    input_dir = Path(INPUT_DOCUMENTS_DIR)
    
    if not vector_store_path.exists():
        print("Vector store not found. Building it from documents...")
        
        # Check if input directory exists and has documents
        if not input_dir.exists():
            print(f"Error: Input directory {INPUT_DOCUMENTS_DIR} does not exist")
            print("Please create the directory and add documents before running the application.")
            raise FileNotFoundError(f"Input directory {INPUT_DOCUMENTS_DIR} not found")
            
        # Get all document paths
        if not input_dir.exists():
            print(f"Error: No documents found in {INPUT_DOCUMENTS_DIR}")
            print("Please add documents to the input directory before running the application.")
            raise FileNotFoundError(f"No documents found in {INPUT_DOCUMENTS_DIR}")
            
        try:
            # Build vector store with document paths
            build_vector_store(
                input_dir_path=input_dir,
                vector_db_path=DEFAULT_VECTOR_DB_PATH,
                embedding_model_path=DEFAULT_EMBEDDING_MODEL_PATH
            )
            print("Vector store built successfully!")
        except Exception as e:
            print(f"Error building vector store: {str(e)}")
            print("Please ensure your documents are valid and try again.")
            raise
    else:
        print(f"Using existing vector store at {DEFAULT_VECTOR_DB_PATH}")

def inspect_vocabulary(embedding_model: CBOWModel) -> Dict[str, Any]:
    """Inspect the vocabulary and word statistics.
    
    Args:
        embedding_model: The CBOW model to inspect
        
    Returns:
        Dictionary containing vocabulary statistics
    """
    if not embedding_model.vocab:
        return {"error": "No vocabulary found. Model may not have been trained."}
    
    # Get word frequencies
    word_freq = embedding_model.word_freq
    
    # Get most common and least common words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_words[:20]
    least_common = sorted_words[-20:]
    
    return {
        "vocabulary_size": len(embedding_model.vocab),
        "most_common_words": most_common,
        "least_common_words": least_common,
        "sample_words": list(embedding_model.vocab)[:50],  # First 50 words
        "word_frequency_stats": {
            "min": min(word_freq.values()),
            "max": max(word_freq.values()),
            "mean": np.mean(list(word_freq.values())),
            "median": np.median(list(word_freq.values()))
        }
    }

def inspect_vector_db() -> Dict[str, Any]:
    """Inspect the vector database.
    
    Returns:
        Dictionary containing vector database statistics
    """
    if not Path(DEFAULT_VECTOR_DB_PATH).exists():
        return {"error": "Vector database not found"}
    
    try:
        vector_db = VectorDB.load(DEFAULT_VECTOR_DB_PATH)
        
        # Calculate vector statistics
        norms = np.linalg.norm(vector_db.vectors, axis=1)
        
        return {
            "num_vectors": len(vector_db.vectors),
            "embedding_dimension": vector_db.embedding_dim,
            "vector_statistics": {
                "min_norm": float(np.min(norms)),
                "max_norm": float(np.max(norms)),
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms))
            },
            "sample_texts": vector_db.texts[:5] if vector_db.texts else []
        }
        
    except Exception as e:
        return {"error": f"Failed to inspect vector database: {str(e)}"}

def print_inspection_results():
    """Print formatted inspection results for both vocabulary and vector database."""
    print("\n=== RAG System Inspection ===\n")
    
    # Check vector database
    print("1. Vector Database Inspection:")
    print("-" * 50)
    db_stats = inspect_vector_db()
    
    if "error" in db_stats:
        print(f"{db_stats['error']}")
    else:
        print(f"Vector database found at: {DEFAULT_VECTOR_DB_PATH}")
        print(f"\nDatabase Statistics:")
        print(f"  • Number of vectors: {db_stats['num_vectors']}")
        print(f"  • Embedding dimension: {db_stats['embedding_dimension']}")
        print("\nVector Statistics:")
        stats = db_stats['vector_statistics']
        print(f"  • Min norm: {stats['min_norm']:.4f}")
        print(f"  • Max norm: {stats['max_norm']:.4f}")
        print(f"  • Mean norm: {stats['mean_norm']:.4f}")
        print(f"  • Std norm: {stats['std_norm']:.4f}")
        
        if db_stats['sample_texts']:
            print("\nSample Texts (first 5):")
            for i, text in enumerate(db_stats['sample_texts'], 1):
                print(f"\n{i}. {text[:200]}...")
    
    # Check vocabulary
    print("\n2. Vocabulary Inspection:")
    print("-" * 50)
    embedding_model = CBOWModel()
    vocab_stats = inspect_vocabulary(embedding_model)
    
    if "error" in vocab_stats:
        print(f"{vocab_stats['error']}")
    else:
        print(f"\nVocabulary Statistics:")
        print(f"  • Vocabulary size: {vocab_stats['vocabulary_size']}")
        
        print("\nMost Common Words:")
        for word, freq in vocab_stats['most_common_words']:
            print(f"  • {word}: {freq}")
        
        print("\nLeast Common Words:")
        for word, freq in vocab_stats['least_common_words']:
            print(f"  • {word}: {freq}")
        
        print("\nWord Frequency Statistics:")
        stats = vocab_stats['word_frequency_stats']
        print(f"  • Min frequency: {stats['min']:.2f}")
        print(f"  • Max frequency: {stats['max']:.2f}")
        print(f"  • Mean frequency: {stats['mean']:.2f}")
        print(f"  • Median frequency: {stats['median']:.2f}")
        
        print("\nSample Words (first 50):")
        print(", ".join(vocab_stats['sample_words']))

if __name__ == "__main__":
    print_inspection_results()
