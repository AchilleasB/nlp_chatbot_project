"""Base class for Word2Vec implementations.

This module provides the base class for Word2Vec implementations, defining the common
interface and functionality shared by different architectures (CBOW, Skip-gram).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter
import re
from tqdm import tqdm
import pickle
from pathlib import Path
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseWord2Vec(ABC):
    """Base class for Word2Vec implementations.
    
    This abstract base class defines the common interface and functionality
    for Word2Vec implementations. It handles vocabulary building, vector
    initialization, and provides utility methods for word-level operations.
    """
    
    def __init__(
        self,
        embedding_dim: int = 300,
        window_size: int = 5,
        min_count: int = 5,
        learning_rate: float = 0.025,
        epochs: int = 5,
        negative_samples: int = 5,
        subsample_threshold: float = 1e-5
    ):
        """Initialize the base Word2Vec model.
        
        Args:
            embedding_dim: Dimension of the word vectors
            window_size: Size of the context window (words on each side)
            min_count: Minimum word frequency to be included in vocabulary
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            negative_samples: Number of negative samples for each positive sample
            subsample_threshold: Threshold for subsampling frequent words
        """
        # Model parameters
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.subsample_threshold = subsample_threshold
        
        # Model state
        self.word_vectors = {}  # Word embeddings (W)
        self.context_vectors = {}  # Context embeddings (W')
        self.vocab = set()  # Vocabulary
        self.word_freq = Counter()  # Word frequencies
        self.word_to_idx = {}  # Word to index mapping
        self.idx_to_word = {}  # Index to word mapping
        self.word_probs = None  # Word probabilities for negative sampling
        self.total_words = 0  # Total word count for probability calculations
        self.is_trained = False
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split into words
        # Keep only alphanumeric characters and basic punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts.
        
        This method:
        1. Counts word frequencies
        2. Filters vocabulary by minimum count
        3. Creates word-to-index mappings
        4. Computes word probabilities for negative sampling
        
        Args:
            texts: List of texts to build vocabulary from
        """
        logger.info("Building vocabulary...")
        
        # Count word frequencies
        for text in tqdm(texts, desc="Processing documents"):
            words = self._preprocess_text(text)
            self.word_freq.update(words)
        
        # Filter vocabulary by minimum count
        self.vocab = {word for word, freq in self.word_freq.items() 
                     if freq >= self.min_count}
        
        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(self.vocab))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Compute word probabilities for negative sampling
        self.total_words = sum(self.word_freq[word] for word in self.vocab)
        self.word_probs = np.array([self.word_freq[word] / self.total_words 
                                  for word in sorted(self.vocab)])
        
        # Raise to 3/4 power as in the original Word2Vec paper
        # This helps reduce the impact of very frequent words
        self.word_probs = self.word_probs ** 0.75
        self.word_probs /= self.word_probs.sum()
        
        logger.info(f"Vocabulary size: {len(self.vocab)} words")
    
    def _initialize_vectors(self):
        """Initialize word and context vectors.
        
        Vectors are initialized using random values from a uniform distribution
        scaled by the embedding dimension. This helps prevent the vectors from
        growing too large during training.
        """
        logger.info("Initializing vectors...")
        
        # Initialize with small random values
        for word in tqdm(self.vocab, desc="Initializing vectors"):
            # Word vectors (W)
            self.word_vectors[word] = np.random.uniform(
                low=-0.5/self.embedding_dim,
                high=0.5/self.embedding_dim,
                size=self.embedding_dim
            )
            
            # Context vectors (W')
            self.context_vectors[word] = np.random.uniform(
                low=-0.5/self.embedding_dim,
                high=0.5/self.embedding_dim,
                size=self.embedding_dim
            )
    
    def _get_context_words(self, words: List[str], idx: int) -> List[str]:
        """Get context words for a given word index.
        
        Args:
            words: List of words
            idx: Index of the target word
            
        Returns:
            List of context words within the window size
        """
        start = max(0, idx - self.window_size)
        end = min(len(words), idx + self.window_size + 1)
        context = words[start:idx] + words[idx+1:end]
        return [w for w in context if w in self.vocab]
    
    def _should_subsample(self, word: str) -> bool:
        """Determine if a word should be subsampled based on its frequency.
        
        This implements the subsampling technique from the Word2Vec paper to
        reduce the impact of very frequent words.
        
        Args:
            word: Word to check
            
        Returns:
            True if the word should be subsampled, False otherwise
        """
        if word not in self.word_freq:
            return True
            
        freq = self.word_freq[word] / self.total_words
        return random.random() < (1 - np.sqrt(self.subsample_threshold / freq))
    
    def _get_negative_samples(self, target_word: str) -> List[str]:
        """Get negative samples for a target word.
        
        Negative sampling helps improve training efficiency by only updating
        a small number of negative examples for each positive example.
        
        Args:
            target_word: The target word to get negative samples for
            
        Returns:
            List of negative sample words
        """
        # Get random indices based on word probabilities
        neg_indices = np.random.choice(
            len(self.vocab),
            size=self.negative_samples,
            p=self.word_probs,
            replace=False
        )
        
        # Convert indices to words, excluding the target word
        neg_words = [self.idx_to_word[idx] for idx in neg_indices 
                    if self.idx_to_word[idx] != target_word]
        
        return neg_words
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """Get vector for a single word.
        
        Args:
            word: Input word
            
        Returns:
            Word vector (zero vector if word not in vocabulary)
        """
        return self.word_vectors.get(word, np.zeros(self.embedding_dim))
    
    def get_most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most similar words to a given word.
        
        Args:
            word: Input word
            top_n: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
            
        Raises:
            KeyError: If word not in vocabulary
        """
        if word not in self.word_vectors:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        word_vec = self.word_vectors[word]
        similarities = []
        
        for other_word, other_vec in self.word_vectors.items():
            if other_word != word:
                # Compute cosine similarity
                similarity = np.dot(word_vec, other_vec) / (
                    np.linalg.norm(word_vec) * np.linalg.norm(other_vec)
                )
                similarities.append((other_word, float(similarity)))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    def save(self, path: str):
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        data = {
            'embedding_dim': self.embedding_dim,
            'window_size': self.window_size,
            'min_count': self.min_count,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'negative_samples': self.negative_samples,
            'subsample_threshold': self.subsample_threshold,
            'word_vectors': self.word_vectors,
            'context_vectors': self.context_vectors,
            'vocab': self.vocab,
            'word_freq': self.word_freq,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_probs': self.word_probs,
            'total_words': self.total_words,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseWord2Vec':
        """Load a model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        model = cls(
            embedding_dim=data['embedding_dim'],
            window_size=data['window_size'],
            min_count=data['min_count'],
            learning_rate=data['learning_rate'],
            epochs=data['epochs'],
            negative_samples=data['negative_samples'],
            subsample_threshold=data['subsample_threshold']
        )
        
        model.word_vectors = data['word_vectors']
        model.context_vectors = data['context_vectors']
        model.vocab = data['vocab']
        model.word_freq = data['word_freq']
        model.word_to_idx = data['word_to_idx']
        model.idx_to_word = data['idx_to_word']
        model.word_probs = data['word_probs']
        model.total_words = data['total_words']
        model.is_trained = data['is_trained']
        
        return model
    
    @abstractmethod
    def train(self, texts: List[str]):
        """Train the model on texts.
        
        Args:
            texts: List of texts to train on
        """
        pass
    
    @abstractmethod
    def _update_vectors(self, *args, **kwargs):
        """Update word and context vectors for a training example.
        
        This method must be implemented by subclasses to define how vectors
        are updated during training.
        """
        pass 