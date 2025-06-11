"""
Text preprocessing and chunking module for the chatbot system.
This module handles document preprocessing, cleaning, and chunking strategies.
"""

import re
import logging
from typing import List, Dict, Optional
from pathlib import Path
from .embeddings import TextChunk
import docx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 500,
                 overlap_size: int = 50):
        """
        Initialize the text preprocessor.
        
        Args:
            min_chunk_size: Minimum number of characters per chunk
            max_chunk_size: Maximum number of characters per chunk
            overlap_size: Number of characters to overlap between chunks
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        # Compile regex pattern for sentence splitting
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        logger.info(f"Initialized TextPreprocessor with chunk size {max_chunk_size} and overlap {overlap_size}")
    
    def _read_docx(self, file_path: Path) -> str:
        """
        Read text from a .docx file.
        
        Args:
            file_path: Path to the .docx file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Reading DOCX file: {file_path}")
        doc = docx.Document(file_path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        logger.info(f"Extracted {len(text)} characters from DOCX")
        return text
    
    def _read_text_file(self, file_path: Path) -> str:
        """
        Read text from a .txt file.
        
        Args:
            file_path: Path to the .txt file
            
        Returns:
            File content
        """
        logger.info(f"Reading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Read {len(text)} characters from text file")
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?-])', r'\1', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex pattern matching.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on sentence boundaries and filter out empty strings
        sentences = [s.strip() for s in self.sentence_pattern.split(text) if s.strip()]
        return sentences
    
    def create_chunks(self, 
                     text: str, 
                     metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        Create chunks from text using a sliding window approach.
        
        Args:
            text: Input text
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        # Clean the text
        text = self.clean_text(text)
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed max size, create a new chunk
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata=metadata or {}
                    ))
                
                # Keep overlap_size characters from the end
                overlap_text = chunk_text[-self.overlap_size:]
                overlap_sentences = []
                overlap_size = 0
                
                # Find sentences that fit in the overlap
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.overlap_size:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add current sentence to chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it meets minimum size
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata=metadata or {}
                ))
        
        return chunks
    
    def process_file(self, 
                    file_path: Path, 
                    metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        Process a file and create chunks.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        logger.info(f"Processing file: {file_path}")
        try:
            # Read file content based on file type
            if file_path.suffix.lower() == '.docx':
                text = self._read_docx(file_path)
            else:
                text = self._read_text_file(file_path)
            
            if not text.strip():
                logger.warning(f"No text content found in {file_path}")
                return []
            
            # Add file metadata
            file_metadata = {
                'filename': file_path.name,
                'file_type': file_path.suffix,
                **(metadata or {})
            }
            
            chunks = self.create_chunks(text, file_metadata)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            return []
    
    def process_directory(self, dir_path: Path, file_pattern: str = "*") -> List[TextChunk]:
        """
        Process all files in a directory matching the pattern.
        
        Args:
            dir_path: Path to the directory
            file_pattern: Base pattern for files to process (default: "*")
            
        Returns:
            List of TextChunk objects from all files
        """
        logger.info(f"Processing directory: {dir_path}")
        all_chunks = []
        
        # Get all files and filter by extension
        all_files = list(dir_path.glob(file_pattern))
        matching_files = [f for f in all_files if f.suffix.lower() in ['.txt', '.docx']]
        logger.info(f"Found {len(matching_files)} matching files: {[f.name for f in matching_files]}")
        
        for file_path in matching_files:
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks 