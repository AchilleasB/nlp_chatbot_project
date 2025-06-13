"""Core functionality for the application."""

import os
import argparse
import subprocess
import shutil
import time
from typing import Tuple, Optional, List
from pathlib import Path

from src.chunking.preprocessing import TextPreprocessor
from src.embeddings import CBOWModel, DocumentEmbedder
from src.vectordb import VectorDB
from src.chat import ChatManager
from src.config import DEFAULT_VECTOR_DB_PATH, DEFAULT_EMBEDDING_MODEL_PATH, INPUT_DOCUMENTS_DIR

def check_ollama_installation() -> Tuple[bool, str]:
    """Check if Ollama is installed and running.
    
    Returns:
        Tuple of (is_installed, status_message)
    """
    # Check if ollama command exists
    if not shutil.which('ollama'):
        return False, "Ollama is not installed. Please install Ollama from https://ollama.ai"
    
    # Check if ollama service is running
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            return True, "Ollama is running"
        else:
            return False, "Ollama service is not running. Please start Ollama using 'ollama serve'"
    except subprocess.TimeoutExpired:
        return False, "Ollama service is not responding. Please check if it's running"
    except Exception as e:
        return False, f"Error checking Ollama status: {str(e)}"

def start_ollama_service() -> Tuple[bool, str]:
    """Start the Ollama service if it's not running.
    
    Returns:
        Tuple of (success, status_message)
    """
    try:
        # Try to start ollama serve in the background
        process = subprocess.Popen(['ollama', 'serve'], 
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        
        # Wait a bit for the service to start
        time.sleep(2)
        
        # Check if it's running now
        is_running, status = check_ollama_installation()
        if is_running:
            return True, "Ollama service started successfully"
        else:
            return False, f"Failed to start Ollama service: {status}"
    except Exception as e:
        return False, f"Error starting Ollama service: {str(e)}"

def get_installed_models() -> Tuple[List[str], bool]:
    """Get list of installed Ollama models.
    
    Returns:
        Tuple of (list of model names, has_models)
    """
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              check=True,
                              timeout=5)
        
        # Parse the output
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # Only header or empty
            return [], False
            
        # Skip header line and parse model names
        model_names = []
        for line in lines[1:]:  # Skip header
            if line.strip():  # Skip empty lines
                # Split by whitespace and get the first column (model name)
                model_name = line.split()[0]
                # Remove :latest suffix if present
                if model_name.endswith(':latest'):
                    model_name = model_name[:-7]
                model_names.append(model_name)
        
        return model_names, bool(model_names)
    except Exception as e:
        print(f"Error getting installed models: {str(e)}")
        return [], False

def run_cli(model_name: str = None):
    """Run the command-line interface for the application.
    
    Args:
        model_name: Optional name of the Ollama model to use. If not provided,
                   will use command line arguments to determine the model.
    """
    parser = argparse.ArgumentParser(description="NLP Chatbot with RAG")
    parser.add_argument("--add-document", help="Path to document to add to database")
    parser.add_argument("--model", default="mistral", help="Name of the Ollama model to use")
    args = parser.parse_args()
    
    # Use provided model_name if given, otherwise use args.model
    model_to_use = model_name if model_name is not None else args.model
    
    # Check Ollama installation and status
    print("\n=== Checking Ollama Status ===")
    is_installed, status = check_ollama_installation()
    if not is_installed:
        print(f"Error: {status}")
        if "not installed" in status.lower():
            print("\nPlease install Ollama from https://ollama.ai")
            return
        elif "not running" in status.lower():
            print("\nAttempting to start Ollama service...")
            success, start_status = start_ollama_service()
            if not success:
                print(f"Error: {start_status}")
                print("\nPlease start Ollama manually using 'ollama serve'")
                return
            print("Success: Ollama service started")
        else:
            return
    
    # Check if the requested model is installed
    print("\n=== Checking Model Availability ===")
    models, has_models = get_installed_models()
    if not has_models:
        print("No models installed yet. Installing default model (mistral)...")
        try:
            subprocess.run(['ollama', 'pull', 'mistral'], check=True)
            print("Successfully installed mistral model")
            models = ['mistral']
            has_models = True
        except Exception as e:
            print(f"Error installing model: {str(e)}")
            print("\nPlease install a model manually using:")
            print("  ollama pull mistral  # Recommended (~4GB)")
            print("  ollama pull llama2   # Alternative (~4GB)")
            print("  ollama pull codellama # For code (~4GB)")
            return
    
    if model_to_use not in models:
        print(f"Model '{model_to_use}' is not installed. Installing...")
        try:
            subprocess.run(['ollama', 'pull', model_to_use], check=True)
            print(f"Successfully installed {model_to_use} model")
        except Exception as e:
            print(f"Error installing model: {str(e)}")
            print(f"\nAvailable models: {', '.join(models)}")
            print("Please choose one of the available models or install a new one using:")
            print("  ollama pull <model_name>")
            return
    
    print(f"\nUsing model: {model_to_use}")
    print("=== Ollama Setup Complete ===\n")
    
    # Initialize components
    embedding_model, document_embedder, vector_db, chat_manager = initialize_components()
    
    # Set the model name in chat manager
    chat_manager.model_name = model_to_use
    
    # Add document if specified
    if args.add_document:
        with open(args.add_document, 'r', encoding='utf-8') as f:
            text = f.read()
        process_document(text, embedding_model, document_embedder, vector_db)
        vector_db.save(DEFAULT_VECTOR_DB_PATH)
        print(f"Document processed and added to database")
        return
    
    # Interactive chat loop
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'quit':
            break
            
        try:
            response = chat_manager.chat(query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error: {str(e)}")

def initialize_components() -> Tuple[CBOWModel, DocumentEmbedder, VectorDB, ChatManager]:
    """Initialize all necessary components for the application."""
    print("\n=== Initializing RAG Components ===")
    
    # Initialize components
    vector_db_path = DEFAULT_VECTOR_DB_PATH
    embedding_model_path = DEFAULT_EMBEDDING_MODEL_PATH

    print(f"\n1. Loading/Creating Embedding Model:")
    print(f"   - Model path: {embedding_model_path}")
    if Path(embedding_model_path).exists():
        print("   - Found existing model, loading...")
        embedding_model = CBOWModel.load(embedding_model_path)
        print(f"   - Model loaded. Vocabulary size: {len(embedding_model.vocab)}")
        print(f"   - Model trained: {embedding_model.is_trained}")
    else:
        print("   - No existing model found, creating new...")
        embedding_model = CBOWModel()
        print("   - New model created (untrained)")
    
    print(f"\n2. Creating Document Embedder:")
    document_embedder = DocumentEmbedder(embedding_model)
    print(f"   - Embedding dimension: {document_embedder.get_embedding_dimension()}")
    
    print(f"\n3. Loading/Creating Vector Database:")
    print(f"   - DB path: {vector_db_path}")
    
    # Check if vector store exists and has content
    vector_store_exists = Path(vector_db_path).exists()
    vector_db = None
    
    if vector_store_exists:
        print("   - Found existing vector store, loading...")
        try:
            vector_db = VectorDB.load(vector_db_path)
            num_vectors = len(vector_db.vectors) if hasattr(vector_db, 'vectors') else 0
            print(f"   - Vector store loaded. Number of vectors: {num_vectors}")
            
            if num_vectors == 0:
                print("   - Warning: Vector store is empty!")
                print("   - Rebuilding vector store from documents...")
                # Rebuild vector store if empty
                embedding_model, document_embedder, vector_db = build_vector_store(
                    input_dir_path=INPUT_DOCUMENTS_DIR,
                    vector_db_path=vector_db_path,
                    embedding_model_path=embedding_model_path,
                    embedding_model=embedding_model,
                    document_embedder=document_embedder,
                    vector_db=vector_db
                )
                print("   - Vector store rebuilt successfully")
        except Exception as e:
            print(f"   - Error loading vector store: {str(e)}")
            print("   - Creating new vector store...")
            vector_db = VectorDB(embedding_dim=document_embedder.get_embedding_dimension())
            print("   - New vector store created (empty)")
    else:
        print("   - No existing vector store found, creating new...")
        vector_db = VectorDB(embedding_dim=document_embedder.get_embedding_dimension())
        print("   - New vector store created (empty)")
    
    # Check if we need to process documents (either no vector store or empty vector store)
    if not vector_store_exists or len(vector_db) == 0:
        print("   - Checking for documents to process...")
        input_dir = Path(INPUT_DOCUMENTS_DIR)
        
        if input_dir.exists() and any(input_dir.iterdir()):
            print(f"   - Found documents in {INPUT_DOCUMENTS_DIR}, processing...")
            try:
                embedding_model, document_embedder, vector_db = build_vector_store(
                    input_dir_path=INPUT_DOCUMENTS_DIR,
                    vector_db_path=vector_db_path,
                    embedding_model_path=embedding_model_path,
                    embedding_model=embedding_model,
                    document_embedder=document_embedder,
                    vector_db=vector_db
                )
                print("   - Vector store built successfully from documents")
            except Exception as e:
                print(f"   - Error building vector store: {str(e)}")
                print("   - Continuing with empty vector store")
        else:
            print(f"   - No documents found in {INPUT_DOCUMENTS_DIR}")
            print("   - Please add documents to enable RAG functionality")
    
    print("\n4. Initializing Chat Manager:")
    chat_manager = ChatManager()
    
    # Set RAG components if vector store has vectors
    if len(vector_db) > 0:
        print("   - Setting up RAG components for chat...")
        chat_manager.set_rag_components(document_embedder, vector_db)
        print("   - RAG components configured")
    else:
        print("   - Warning: Vector store is empty, chat will run without RAG")
    
    print("\n=== RAG Components Initialized ===\n")
    return embedding_model, document_embedder, vector_db, chat_manager


def build_vector_store(
    input_dir_path: str,
    vector_db_path: str = DEFAULT_VECTOR_DB_PATH,
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
    embedding_model: Optional[CBOWModel] = None,
    document_embedder: Optional[DocumentEmbedder] = None,
    vector_db: Optional[VectorDB] = None
) -> Tuple[CBOWModel, DocumentEmbedder, VectorDB]:
    """Process documents and build the vector store.
    
    Args:
        input_dir_path: Directory containing documents to process
        vector_db_path: Path to save the vector store
        embedding_model_path: Path to save the embedding model
        embedding_model: Optional pre-initialized embedding model
        document_embedder: Optional pre-initialized document embedder
        vector_db: Optional pre-initialized vector database
        
    Returns:
        Tuple of (embedding_model, document_embedder, vector_db)
    """
    print(f"\nBuilding vector store from documents in {input_dir_path}")
    
    # Initialize components if not provided
    if embedding_model is None:
        print("Creating new embedding model...")
        embedding_model = CBOWModel()
    if document_embedder is None:
        document_embedder = DocumentEmbedder(embedding_model)
    if vector_db is None:
        vector_db = VectorDB(document_embedder.get_embedding_dimension())
    
    # Initialize text preprocessor with our components
    preprocessor = TextPreprocessor(
        embedding_model=embedding_model,
        document_embedder=document_embedder,
        vector_db=vector_db
    )
    
    # Process all documents in the input directory
    print("Processing documents...")
    all_chunks = preprocessor.process_directory(input_dir_path)
    
    if not all_chunks:
        print("No chunks were generated from documents!")
        return embedding_model, document_embedder, vector_db
    
    print(f"Successfully processed {len(all_chunks)} chunks from documents")
    
    # Ensure the embedding model is trained
    if not embedding_model.is_trained:
        print("Training embedding model on all chunks...")
        embedding_model.train(all_chunks)
        print("Embedding model training completed!")
    
    # Save vector store and embedding model
    print("Saving components...")
    os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
    preprocessor.vector_db.save(vector_db_path)
    print(f"Vector store saved to {vector_db_path}")
    
    os.makedirs(os.path.dirname(embedding_model_path), exist_ok=True)
    embedding_model.save(embedding_model_path)
    print(f"Embedding model saved to {embedding_model_path}")
    
    print(f"Vector store built successfully with {len(preprocessor.vector_db)} vectors")
    
    # Return the populated vector_db from preprocessor
    return embedding_model, document_embedder, preprocessor.vector_db

    # The process_document is only used when explicitly adding a new document to the python app.py via the --add-document flag followed by a path to the document
def process_document(text: str, embedding_model: CBOWModel, document_embedder: DocumentEmbedder, vector_db: VectorDB) -> None:
    """Process a document and add it to the vector database.
    
    Args:
        text: The document text to process
        embedding_model: The CBOW model for word embeddings
        document_embedder: The document embedder for converting text to vectors
        vector_db: The vector database to add to
    """
    # Use the provided components instead of creating new ones
    preprocessor = TextPreprocessor(
        embedding_model=embedding_model,
        document_embedder=document_embedder,
        vector_db=vector_db
    )
    chunks = preprocessor.process_text(text)
    