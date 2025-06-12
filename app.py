#!/usr/bin/env python3
"""
Main application entry point for the NLP chatbot system.
This module provides the command-line interface and orchestrates the components.
"""

import sys
import subprocess
import shutil
from pathlib import Path
import typer
from src.preprocessing import TextPreprocessor
from src.vector_db import VectorDB
from src.chat import ChatBot

app = typer.Typer()

def is_ollama_installed() -> bool:
    """Check if Ollama is installed on the system."""
    return shutil.which("ollama") is not None

def start_ollama() -> bool:
    """Start the Ollama service if it's not running."""
    try:
        # Try to start Ollama in the background
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        # Wait a bit for the service to start
        import time
        time.sleep(2)
        return True
    except Exception as e:
        print(f"Error starting Ollama: {str(e)}")
        return False

def check_ollama() -> bool:
    """Check if Ollama is installed and running, start it if needed."""
    # First check if Ollama is installed
    if not is_ollama_installed():
        print("Error: Ollama is not installed on your system.")
        print("Please visit https://ollama.com/ to download and install Ollama.")
        print("After installation, restart this application.")
        return False
    
    # Check if Ollama is running
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return True
    except:
        # Ollama is not running, try to start it
        print("Ollama is not running. Attempting to start it...")
        if start_ollama():
            # Verify it started successfully
            try:
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    print("Successfully started Ollama.")
                    return True
            except:
                pass
        print("Failed to start Ollama. Please start it manually and try again.")
        return False

def print_welcome(vector_db) -> None:
    """Print welcome message."""
    print("\n=== Welcome to the NLP Chatbot! ===\n")
    print("You can ask questions about your project documents, and I'll try to help you find relevant information.")
    print("\nSpecial commands:")
    print("- /help  - Show available commands")
    print("- /reset - Reset conversation history")
    print("- /load  - Load documents from data/raw directory")
    print("- /stats - Show vector database statistics")
    print("- /exit  - Exit the chatbot")
    
    # Print warning if no documents are loaded
    if not vector_db.chunks:
        print("\n! Note: No documents are currently loaded in the system.")
        print("You can add documents to the data/raw directory")
        print("and use the /load command to load them.\n")

@app.command()
def main(
    data_dir: Path = typer.Option(
        "data/raw",
        help="Directory containing the project documents"
    ),
    model: str = typer.Option(
        "mistral",
        help="Ollama model to use"
    ),
    chunk_size: int = typer.Option(
        500,
        help="Maximum chunk size in characters"
    ),
    overlap: int = typer.Option(
        50,
        help="Overlap size between chunks in characters"
    )
):
    """
    Start the NLP chatbot system.
    """
    # Check if Ollama is running
    if not check_ollama():
        print("Error: Ollama is not running.")
        print("Visit https://ollama.com/ for installation instructions.")
        sys.exit(1)
    
    # Initialize components
    print("\nInitializing NLP Chatbot System")
    
    # Create vector database
    print("\nInitializing vector database...")
    vector_db = VectorDB()
    
    # Check if we need to process documents
    if not vector_db.chunks:
        if not data_dir.exists():
            print(f"Warning: Data directory '{data_dir}' does not exist.")
            print("Please add your documents to the data/raw directory and restart the application.")
            if not typer.confirm("Do you want to continue without documents?"):
                sys.exit(0)
        else:
            # Process documents
            print("\nProcessing documents...")
            preprocessor = TextPreprocessor(
                max_chunk_size=chunk_size,
                overlap_size=overlap
            )
            
            chunks = preprocessor.process_directory(data_dir)
            if not chunks:
                print("No documents found in the data directory.")
                print("Please add your documents to the data/raw directory and restart the application.")
                if not typer.confirm("Do you want to continue without documents?"):
                    sys.exit(0)
            else:
                print(f"Processed {len(chunks)} chunks from documents.")
                
                # Add chunks to vector database
                print("\nCreating embeddings...")
                vector_db.add_chunks(chunks)
                print("Embeddings created and stored.")
    else:
        print("Using existing vector database.")
    
    # Initialize chatbot
    print("\nInitializing chatbot...")
    chatbot = ChatBot(vector_db, model_name=model)
    print("Chatbot initialized.")
    
    # Print welcome message
    print_welcome(vector_db)
    
    # Start chat loop
    while True:
        try:
            # Get user input
            query = input("\nYou: ")
            
            # Handle commands
            if query.startswith("/"):
                response = chatbot.handle_command(query)
                if response == "exit":
                    break
                if response:
                    print(f"\nAssistant: {response}")
                continue
            
            # Process query
            response = chatbot.process_query(query)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    app() 