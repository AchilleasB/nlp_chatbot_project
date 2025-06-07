"""Main entry point for the NLP Chatbot application."""

import sys
import argparse
from pathlib import Path

from src.core import initialize_components, run_cli
from src.utils import ensure_directories_exist, ensure_vector_store_exists
from src.config import DEFAULT_VECTOR_DB_PATH, INPUT_DOCUMENTS_DIR

def is_running_with_streamlit() -> bool:
    """Check if the application is being run with Streamlit.
    
    Returns:
        bool: True if running with Streamlit, False otherwise
    """
    return 'streamlit' in sys.modules

def main():
    """Main entry point for the application.
    
    This function determines whether to run the CLI or Streamlit interface
    based on how the application is being run.
    """
    if is_running_with_streamlit():
        # Import and run Streamlit app
        from streamlit_app import main as run_streamlit
        run_streamlit()
    else:
        # Parse command line arguments for CLI mode
        parser = argparse.ArgumentParser(description='NLP Chatbot CLI')
        parser.add_argument('--model', type=str, default='mistral',
                          help='Name of the Ollama model to use (default: mistral)')
        args = parser.parse_args()
        
        # Ensure required directories exist
        ensure_directories_exist()
        
        # Run CLI version
        run_cli(args.model)

if __name__ == "__main__":
    main()
        
