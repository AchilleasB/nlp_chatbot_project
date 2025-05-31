"""Word2Vec implementation using CBOW (Continuous Bag of Words) architecture.

This implementation follows the CBOW model architecture as described in the original Word2Vec paper:
"Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013).

The CBOW model predicts a target word given its context words, which helps capture
the meaning of words based on their surrounding context. This implementation includes
negative sampling for efficient training and better quality word embeddings.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import random
import logging
from .base_model import BaseWord2Vec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CBOWModel(BaseWord2Vec):
    """CBOW (Continuous Bag of Words) Word2Vec implementation with negative sampling.
    
    The CBOW model learns word embeddings by predicting a target word from its context words.
    For each target word in the text, it uses the surrounding context words within a
    specified window size to predict the target word. This helps capture the meaning
    of words based on their context.
    """
    
    def _get_context_vector(self, context_words: List[str]) -> np.ndarray:
        """Get the average vector of context words.
        
        Args:
            context_words: List of context words
            
        Returns:
            Average vector of context words
        """
        if not context_words:
            return np.zeros(self.embedding_dim)
            
        # Average the word vectors of context words
        context_vectors = [self.word_vectors[word] for word in context_words]
        return np.mean(context_vectors, axis=0)
    
    def _update_vectors(self, context_words: List[str], target_word: str, 
                       negative_words: List[str], is_positive: bool = True):
        """Update word and context vectors for a training example.
        
        Args:
            context_words: List of context words
            target_word: The target word
            negative_words: List of negative sample words
            is_positive: Whether this is a positive or negative example
        """
        # Get context vector (average of context word vectors)
        context_vector = self._get_context_vector(context_words)
        
        # Forward pass
        score = np.dot(context_vector, self.context_vectors[target_word])
        pred = 1 / (1 + np.exp(-score))  # sigmoid
        
        # Compute gradient
        gradient = (pred - (1 if is_positive else 0)) * self.learning_rate
        
        # Update context vectors for each context word
        for context_word in context_words:
            self.word_vectors[context_word] -= gradient * self.context_vectors[target_word]
        
        # Update target word vector
        self.context_vectors[target_word] -= gradient * context_vector
    
    def train(self, texts: List[str]):
        """Train the CBOW model on texts.
        
        Args:
            texts: List of texts to train on
        """
        # Build vocabulary and initialize vectors (from base class)
        self._build_vocabulary(texts)
        self._initialize_vectors()
        
        logger.info("Training CBOW model...")
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Process each text
            for text in tqdm(texts, desc="Training"):
                words = self._preprocess_text(text)
                self._train_cbow(words)
            
            # Shuffle texts for next epoch
            random.shuffle(texts)
        
        self.is_trained = True
        logger.info("Training completed!")
    
    def _train_cbow(self, words: List[str]):
        """Train using CBOW architecture with negative sampling.
        
        Args:
            words: List of words to train on
        """
        for i, target_word in enumerate(words):
            # Skip if word not in vocabulary or should be subsampled
            if target_word not in self.vocab or self._should_subsample(target_word):
                continue
                
            # Get context words
            context_words = self._get_context_words(words, i)
            if not context_words:
                continue
            
            # Get negative samples
            negative_words = self._get_negative_samples(target_word)
            
            # Update for positive example (target word)
            self._update_vectors(context_words, target_word, negative_words, True)
            
            # Update for negative examples
            for neg_word in negative_words:
                self._update_vectors(context_words, neg_word, negative_words, False)
    
    def predict_word(self, context_words: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        """Predict the most likely word given context words.
        
        Args:
            context_words: List of context words
            top_n: Number of predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        # Get context vector
        context_vector = self._get_context_vector(context_words)
        
        # Compute scores for all words in vocabulary
        scores = []
        for word in self.vocab:
            score = np.dot(context_vector, self.context_vectors[word])
            prob = 1 / (1 + np.exp(-score))  # sigmoid
            scores.append((word, float(prob)))
        
        # Return top N predictions
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

