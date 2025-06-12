import sys
from pathlib import Path
import typer
from core.ollama import check_ollama, cleanup_ollama
from core.startup import initialize_chatbot
from src.vector_db import VectorDB

app = typer.Typer()

@app.command()
def main(
    data_dir: Path = typer.Option("data/raw", help="Directory containing the project documents"),
    model: str = typer.Option("mistral", help="Ollama model to use"),
    chunk_size: int = typer.Option(500, help="Maximum chunk size in characters"),
    overlap: int = typer.Option(50, help="Overlap size between chunks in characters")
):
    """Start the NLP chatbot system."""
    try:
        if not check_ollama():
            sys.exit(1)

        chatbot, vector_db = initialize_chatbot(data_dir, model, chunk_size, overlap)

        print("\n=== Welcome to the NLP Chatbot! ===\n")
        print("Commands: /help /reset /load /stats /exit")
        if not vector_db.chunks:
            print("\n! No documents loaded.")

        while True:
            query = input("\nYou: ").strip()
            if not query:
                continue

            if query.startswith("/"):
                response = chatbot.handle_command(query)
                if response == "exit":
                    cleanup_ollama()
                    break
                if response:
                    print(f"\nAssistant: {response}")
                continue

            response = chatbot.process_query(query)
            print(f"\nAssistant: {response}")

    except KeyboardInterrupt:
        print("\nExiting...")
        cleanup_ollama()
    except Exception as e:
        print(f"\nFatal error: {e}")
        cleanup_ollama()
        sys.exit(1)
