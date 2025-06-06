"""Main entry point for the NLP Chatbot application."""

import os
from pathlib import Path

from src.core import run_cli
from src.utils import ensure_directories_exist, ensure_vector_store_exists, print_inspection_results

if __name__ == "__main__":
    # Ensure required directories exist
    ensure_directories_exist()
    
    # Print current state
    print_inspection_results()
    
    # Ensure vector store exists before starting the app
    ensure_vector_store_exists()
    
    # Run the CLI
    run_cli()
        
