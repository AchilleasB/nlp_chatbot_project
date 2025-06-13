"""Core functionality for the application."""

import os
import argparse
from typing import Tuple, Optional, List
from pathlib import Path

from src.chunking.preprocessing import TextPreprocessor
from src.embeddings import CBOWModel, DocumentEmbedder
from src.vectordb import VectorDB
from src.chat import ChatManager
from src.config import DEFAULT_VECTOR_DB_PATH, DEFAULT_EMBEDDING_MODEL_PATH, INPUT_DOCUMENTS_DIR    

def run_cli():
    """Run the command-line interface for the application."""
    parser = argparse.ArgumentParser(description="NLP Chatbot with RAG")
    parser.add_argument("--add-document", help="Path to document to add to database")
    parser.add_argument("--model", default="mistral", help="Name of the Ollama model to use")
    args = parser.parse_args()
    
    # Initialize components
    embedding_model, document_embedder, vector_db, chat_manager = initialize_components()
    
    # Add document if specified
    if args.add_document:
        with open(args.add_document, 'r', encoding='utf-8') as f:
            text = f.read()
        process_document(text, embedding_model, document_embedder, vector_db)
        vector_db.save(DEFAULT_VECTOR_DB_PATH)
        print(f"Document processed and added to database")
        return
    
    # Interactive chat loop
    print(f"Using model: {args.model}")
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
    