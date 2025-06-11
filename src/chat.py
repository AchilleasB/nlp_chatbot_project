"""
Chat module for handling conversations and LLM integration.
This module implements the chatbot interface and manages conversation context.
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
from .vector_db import VectorDB
from .embeddings import TextChunk
from .preprocessing import TextPreprocessor

class ChatBot:
    def __init__(self, 
                 vector_db: VectorDB,
                 model_name: str = "mistral",
                 ollama_url: str = "http://localhost:11434",
                 context_window: int = 5):
        """
        Initialize the chatbot.
        
        Args:
            vector_db: VectorDB instance for retrieval
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama API
            context_window: Number of previous messages to keep in context
        """
        self.vector_db = vector_db
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.context_window = context_window
        self.conversation_history: List[Dict] = []
    
    def _format_context(self, chunks: List[Tuple[TextChunk, float]]) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            chunks: List of (chunk, similarity) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for chunk, similarity in chunks:
            # Add metadata if available
            metadata_str = ""
            if chunk.metadata:
                metadata_str = f" [Source: {chunk.metadata.get('filename', 'Unknown')}]"
            
            context_parts.append(f"Context{metadata_str} (Relevance: {similarity:.2f}):\n{chunk.text}\n")
        
        return "\n".join(context_parts)
    
    def _format_prompt(self, 
                      query: str, 
                      context: str, 
                      conversation_history: List[Dict]) -> str:
        """
        Format the prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous conversation messages
            
        Returns:
            Formatted prompt string
        """
        # System message
        prompt = """
        You are a helpful AI assistant that answers questions based on the provided context.
        Your responses should be:
        1. Accurate and based only on the given context
        2. Clear and concise
        3. Professional and informative
        4. If the context doesn't contain relevant information, say so

        Context:
        {context}

        Previous conversation:
        {history}

        User: {query}
        """
        
        # Format conversation history
        history_str = ""
        for msg in conversation_history[-self.context_window:]:
            role = msg["role"]
            content = msg["content"]
            history_str += f"{role}: {content}\n"
        
        return prompt.format(
            context=context,
            history=history_str,
            query=query
        )
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call the Ollama API to generate a response.
        
        Args:
            prompt: Formatted prompt for the LLM
            
        Returns:
            Generated response
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error calling Ollama API: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query string
            
        Returns:
            Generated response
        """
        # Retrieve relevant context
        chunks = self.vector_db.search(query, top_k=3)
        context = self._format_context(chunks)
        
        # Format prompt with context and conversation history
        prompt = self._format_prompt(query, context, self.conversation_history)
        
        # Generate response
        response = self._call_ollama(prompt)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
    
    def save_conversation(self, file_path: Path) -> None:
        """
        Save the conversation history to a file.
        
        Args:
            file_path: Path to save the conversation
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_conversation(self, file_path: Path) -> None:
        """
        Load a conversation history from a file.
        
        Args:
            file_path: Path to load the conversation from
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)
    
    def handle_command(self, command: str) -> Optional[str]:
        """
        Handle special commands.
        
        Args:
            command: Command string
            
        Returns:
            Response message or None if not a command
        """
        if command == "/help":
            return """
Available commands:
/help  - Show this help message
/reset - Reset conversation history
/load  - Load documents from data/raw directory
/stats - Show vector database statistics
/exit  - Exit the chatbot
            """
        
        elif command == "/reset":
            self.reset_conversation()
            return "Conversation history has been reset."
        
        elif command == "/load":
            return self.load_documents()
        
        elif command == "/stats":
            stats = self.vector_db.get_stats()
            return f"""
Vector Database Statistics:
- Number of chunks: {stats['num_chunks']}
- Embedding dimension: {stats['embedding_dimension']}
- Storage path: {stats['storage_path']}
- Total size: {stats['total_size_mb']:.2f} MB
            """
        
        elif command == "/exit":
            return "exit"
        
        return None

    def load_documents(self, data_dir: Path = Path("data/raw")) -> str:
        """
        Load documents from the specified directory.
        
        Args:
            data_dir: Path to the directory containing documents
            
        Returns:
            Status message
        """
        if not data_dir.exists():
            return f"Error: Data directory '{data_dir}' does not exist."
        
        # Get existing files before processing
        existing_files = self.vector_db._get_existing_files()
        
        # Process documents
        preprocessor = TextPreprocessor()
        chunks = preprocessor.process_directory(data_dir)
        
        if not chunks:
            return "No documents found in the data directory."
        
        # Get new files
        new_files = {chunk.metadata.get('filename') for chunk in chunks if chunk.metadata}
        
        # Add chunks to vector database
        self.vector_db.add_chunks(chunks)
        
        # Prepare status message
        updated_files = new_files.intersection(existing_files)
        added_files = new_files - existing_files
        
        status_parts = []
        if added_files:
            status_parts.append(f"Added {len(added_files)} new document(s): {', '.join(added_files)}")
        if updated_files:
            status_parts.append(f"Updated {len(updated_files)} existing document(s): {', '.join(updated_files)}")
        
        return "\n".join(status_parts) if status_parts else "No changes made to the database." 