import sys
import typer
from src.preprocessing import TextPreprocessor
from src.vector_db import VectorDB
from src.chat import ChatBot
from core.ollama import cleanup_ollama

def initialize_chatbot(data_dir, model, chunk_size, overlap):
    vector_db = VectorDB()

    if not vector_db.chunks:
        if not data_dir.exists():
            print(f"Directory '{data_dir}' does not exist.")
            if not typer.confirm("Continue without documents?"):
                cleanup_ollama()
                sys.exit(0)
        else:
            preprocessor = TextPreprocessor(chunk_size, overlap)
            chunks = preprocessor.process_directory(data_dir)
            if not chunks:
                print("No valid documents found.")
                if not typer.confirm("Continue without documents?"):
                    cleanup_ollama()
                    sys.exit(0)
            else:
                print(f"Loaded {len(chunks)} chunks.")
                vector_db.add_chunks(chunks)

    chatbot = ChatBot(vector_db, model)
    return chatbot, vector_db
