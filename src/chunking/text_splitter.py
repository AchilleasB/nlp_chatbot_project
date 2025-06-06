"""Custom text splitter implementation."""

from typing import List, Set, Tuple, Dict, Any
import re
from dataclasses import dataclass
from collections import defaultdict

from src.models.document import DocumentSection

class TextSplitter:
    """Custom text splitter for document chunking."""
    
    def __init__(
        self,
        chunk_size: int = 1000,  # Increased chunk size
        chunk_overlap: int = 200,  # Increased chunk overlap    
        min_chunk_size: int = 300,  # Increased minimum chunk size
        max_title_length: int = 100  # Increased max title length
    ):
        """Initialize the text splitter.
        
        Args:
            chunk_size: Target size of each chunk (default: 1000)
            min_chunk_size: Minimum size for a chunk to be considered valid (default: 300)
            max_title_length: Maximum length for a line to be considered a title (default: 100)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_title_length = max_title_length
        
        # Common metadata patterns
        self.metadata_patterns = [
            r'^page\s+\d+$',  # Page numbers
            r'^\d+$',  # Standalone numbers
            r'^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$',  # Dates
            r'^©\s+\d{4}',  # Copyright notices
            r'^confidential',  # Confidentiality notices
            r'^draft',  # Draft notices
            r'^version\s+\d+',  # Version numbers
        ]
        
        # Title patterns with improved detection
        self.title_patterns = [
            r'^[A-Z][A-Z\s]{5,}$',  # ALL CAPS titles
            r'^[IVX]+\.\s+[A-Z]',  # Roman numeral sections
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case lines
            r'^(?:Abstract|Introduction|Methodology|Results|Conclusion|References|Appendix)',  # Common section headers
            r'^[A-Z][a-zA-Z\s]{3,}:$',  # Headers ending with colon
        ]
        
        # Section type patterns
        self.section_patterns = {
            'introduction': r'^(?:Introduction|Background|Overview)',
            'methodology': r'^(?:Methodology|Methods|Approach|Implementation)',
            'results': r'^(?:Results|Findings|Analysis)',
            'conclusion': r'^(?:Conclusion|Summary|Discussion)',
            'references': r'^(?:References|Bibliography)',
            'appendix': r'^(?:Appendix|Appendices)',
        }
    
    def _is_metadata(self, text: str) -> bool:
        """Check if text is metadata."""
        text = text.strip().lower()
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in self.metadata_patterns)
    
    def _is_title(self, text: str) -> bool:
        """Check if text is a title with improved detection."""
        if len(text) > self.max_title_length:
            return False
        text = text.strip()
        return any(re.match(pattern, text) for pattern in self.title_patterns)
    
    def _get_section_type(self, text: str) -> str:
        """Determine the type of section based on its title."""
        text = text.strip()
        for section_type, pattern in self.section_patterns.items():
            if re.match(pattern, text, re.IGNORECASE):
                return section_type
        return "content"
    
    def _get_section_level(self, text: str) -> int:
        """Determine the section level of a line with improved detection."""
        text = text.strip()
        if re.match(r'^[IVX]+\.', text):
            return 1
        if re.match(r'^\d+\.', text):
            return 2
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text):
            return 3
        if re.match(r'^(?:Abstract|Introduction|Methodology|Results|Conclusion|References|Appendix)', text, re.IGNORECASE):
            return 1
        return 0
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving important punctuation."""
        # Pattern to match sentence endings while preserving abbreviations
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_sections(self, text: str) -> List[DocumentSection]:
        """Split text into document sections with improved structure preservation."""
        lines = text.split('\n')
        sections = []
        current_section = []
        current_level = 0
        current_type = "content"
        page_num = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for page breaks
            if re.match(r'^page\s+\d+$', line, re.IGNORECASE):
                page_num = int(re.search(r'\d+', line).group())
                continue
            
            # Handle titles and metadata
            if self._is_metadata(line):
                if current_section:
                    sections.append(DocumentSection(
                        content='\n'.join(current_section),
                        section_level=current_level,
                        section_type=current_type,
                        page_number=page_num
                    ))
                    current_section = []
                sections.append(DocumentSection(
                    content=line,
                    is_metadata=True,
                    page_number=page_num
                ))
                continue
            
            if self._is_title(line):
                if current_section:
                    sections.append(DocumentSection(
                        content='\n'.join(current_section),
                        section_level=current_level,
                        section_type=current_type,
                        page_number=page_num
                    ))
                    current_section = []
                current_level = self._get_section_level(line)
                current_type = self._get_section_type(line)
                sections.append(DocumentSection(
                    content=line,
                    is_title=True,
                    section_level=current_level,
                    section_type=current_type,
                    page_number=page_num
                ))
                continue
            
            current_section.append(line)
        
        # Add the last section
        if current_section:
            sections.append(DocumentSection(
                content='\n'.join(current_section),
                section_level=current_level,
                section_type=current_type,
                page_number=page_num
            ))
        
        return sections
    
    def _create_chunks_from_sections(
        self,
        sections: List[DocumentSection],
        seen_chunks: Set[str]
    ) -> List[str]:
        """Create chunks from document sections with improved content grouping."""
        chunks = []
        current_chunk = []
        current_length = 0
        current_section_type = "content"
        
        for section in sections:
            if section.is_metadata:
                continue
            
            # Start new chunk for new section types
            if section.section_type != current_section_type and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if chunk_text not in seen_chunks and len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                    seen_chunks.add(chunk_text)
                current_chunk = []
                current_length = 0
                current_section_type = section.section_type
            
            # Always include titles with their content
            if section.is_title:
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if chunk_text not in seen_chunks and len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                        seen_chunks.add(chunk_text)
                current_chunk = [section.content]
                current_length = len(section.content)
                continue
            
            # Split content into sentences for better chunking
            sentences = self._split_into_sentences(section.content)
            for sentence in sentences:
                # If adding this sentence would exceed chunk size, save current chunk
                if current_length + len(sentence) + 1 > self.chunk_size and current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if chunk_text not in seen_chunks and len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                        seen_chunks.add(chunk_text)
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
        
        # Add the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text not in seen_chunks and len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
                seen_chunks.add(chunk_text)
        
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while preserving document structure.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split into sections with metadata
        sections = self._split_into_sections(text)
        
        # Create chunks from sections
        seen_chunks = set()
        chunks = self._create_chunks_from_sections(sections, seen_chunks)
        
        return chunks 