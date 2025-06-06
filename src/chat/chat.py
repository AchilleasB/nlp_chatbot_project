"""Chat module implementing RAG-based chat functionality."""

from typing import List, Dict, Any, Optional, Tuple
import ollama
from datetime import datetime
import numpy as np
import logging
import re

from ..config import DEFAULT_MODEL_NAME
from ..models.message import Message
from .exceptions import ChatError, ModelError, ContextError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatManager:
    """Manages chat interactions using RAG."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_context_chunks: int = 3,
        min_relevance_score: float = 0.7,
        max_history_messages: int = 4,
        max_context_length: int = 2000,
        context_relevance_threshold: float = 0.8
    ):
        """Initialize the chat manager.
        
        Args:
            model_name: Name of the Ollama model to use for chat
            num_context_chunks: Number of context chunks to retrieve
            min_relevance_score: Minimum relevance score for context chunks
            max_history_messages: Maximum number of messages to keep in history
            max_context_length: Maximum length of combined context
            context_relevance_threshold: Threshold for deciding whether to use RAG
        """
        self.model_name = model_name
        self.num_context_chunks = num_context_chunks
        self.min_relevance_score = min_relevance_score
        self.max_history_messages = max_history_messages
        self.max_context_length = max_context_length
        self.context_relevance_threshold = context_relevance_threshold
        self.conversation_history: List[Message] = []
        self._rag_components = None  # Will be set when RAG is available

    def chat(self, message: str) -> str:
        """Process a chat message and return a response.
        
        Args:
            message: User message
            
        Returns:
            Model response
            
        Raises:
            ChatError: If chat processing fails
            ModelError: If model interaction fails
        """
        try:
            # Try to get relevant context if RAG is available
            logger.info("Processing chat message...")
            context_result = self._get_relevant_context(message)
            
            if context_result:
                context, max_score = context_result
                logger.info(f"Using RAG-enhanced response (relevance score: {max_score:.4f})")
            else:
                context = None
                logger.info("Using base model response (no RAG)")
            
            # Prepare prompt with context and history
            prompt = self._prepare_prompt(message, context)
            
            # Get model response
            response = self._get_model_response(prompt)
            
            # Update conversation history
            self._update_history(message, response, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            raise ChatError(f"Failed to process chat message: {str(e)}")

    def _get_relevant_context(self, query: str) -> Optional[Tuple[str, float]]:
        """Get relevant context for a query if RAG is available.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (combined context, max relevance score) or None if RAG not available
        """
        if not self._rag_components:
            logger.info("RAG not available - no context will be used")
            return None
            
        document_embedder, vector_db = self._rag_components
        logger.info("Attempting to retrieve context using RAG...")
        
        try:
            # Get query embedding
            logger.info("Generating embedding for query...")
            query_embedding = document_embedder.get_embeddings([query])[0]
            
            # Search for relevant chunks
            results = vector_db.similarity_search(
                query_embedding,
                k=self.num_context_chunks

            )
            
            if not results:
                logger.warning("No relevant chunks found in vector store")
                return None
            
            # Get the highest relevance score
            max_score = max(score for _, score in results)
            logger.info(f"Highest relevance score: {max_score:.4f}")
            
            # If the highest score is below threshold, don't use RAG
            if max_score < self.context_relevance_threshold:
                logger.info(f"Highest relevance score {max_score:.4f} below threshold {self.context_relevance_threshold}, not using RAG")
                return None
            
            context_chunks = []
            total_length = 0
            
            logger.info("Selecting top most relevant chunks for response...")
            for text, score in sorted(results, key=lambda x: x[1], reverse=True)[:self.num_context_chunks]: 
                chunk_length = len(text)
                if total_length + chunk_length > self.max_context_length:
                    break
                    
                context_chunks.append(f"[Relevance: {score:.3f}] {text}")
                total_length += chunk_length
                logger.debug(f"Added chunk with score {score:.3f}")
            
            if not context_chunks:
                logger.warning("No chunks met the relevance threshold")
                return None
            
            logger.info(f"Using {len(context_chunks)} most relevant chunks for response (total length: {total_length})")
            return "\n\n".join(context_chunks), max_score
            
        except Exception as e:
            logger.error(f"Context retrieval error: {str(e)}")
            return None

    def _prepare_prompt(self, message: str, context: Optional[str] = None) -> str:
        """Prepare the prompt for the model.
        
        Args:
            message: User message
            context: Retrieved context, if available
            
        Returns:
            Formatted prompt
        """
        # Start with system message
        if context:
            prompt = """You are a helpful AI assistant with access to relevant context. Your task is to provide a detailed and comprehensive response based on the provided context.

Guidelines for your response:
1. Use the context information to provide specific details and examples
2. If the context contains multiple relevant pieces, combine them into a coherent response
3. If there are any contradictions in the context, acknowledge them
4. If the context is insufficient, say so but still provide what information you can
5. Include specific details, numbers, or examples from the context when available

Context (with relevance scores):
"""
            prompt += f"{context}\n\n"
        else:
            prompt = "You are a helpful AI assistant. Answer the user's question:\n\n"
        
        # Add conversation history
        if self.conversation_history:
            prompt += "Previous conversation:\n"
            for msg in self.conversation_history[-self.max_history_messages:]:
                role = "User" if msg.role == "user" else "Assistant"
                prompt += f"{role}: {msg.content}\n"
            prompt += "\n"
        
        # Add current message
        prompt += f"User: {message}\nAssistant:"
        
        return prompt
    
    def _get_model_response(self, prompt: str) -> str:
        """Get response from the model.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Model response
            
        Raises:
            ModelError: If model interaction fails
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            return response['response']
            
        except Exception as e:
            logger.error(f"Model error: {str(e)}")
            raise ModelError(f"Failed to get model response: {str(e)}")
    
    def _update_history(self, message: str, response: str, context: Optional[str] = None):
        """Update conversation history.
        
        Args:
            message: User message
            response: Model response
            context: Retrieved context, if any
        """
        # Add new messages
        self.conversation_history.append(Message(
            role="user",
            content=message,
            timestamp=datetime.now()
        ))
        self.conversation_history.append(Message(
            role="assistant",
            content=response,
            timestamp=datetime.now(),
            context_used=[context] if context else None
        ))
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_messages * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_messages * 2):]
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation.
        
        Returns:
            Dictionary containing conversation statistics
        """
        if not self.conversation_history:
            return {"message_count": 0}
        
        return {
            "message_count": len(self.conversation_history),
            "user_messages": len([m for m in self.conversation_history if m.role == "user"]),
            "assistant_messages": len([m for m in self.conversation_history if m.role == "assistant"]),
            "start_time": self.conversation_history[0].timestamp,
            "last_message_time": self.conversation_history[-1].timestamp,
            "average_context_length": np.mean([
                len(m.context_used) if m.context_used else 0 
                for m in self.conversation_history 
                if m.role == "assistant"
            ]) if any(m.context_used for m in self.conversation_history) else 0
        } 
    
    def set_rag_components(self, document_embedder, vector_db):
        """Set the RAG components for context retrieval.
        
        Args:
            document_embedder: Document embedder for converting text to vectors
            vector_db: Vector database for storing and retrieving embeddings
        """
        self._rag_components = (document_embedder, vector_db)
    