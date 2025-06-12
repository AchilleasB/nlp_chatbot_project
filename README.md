# NLP Project Chatbot

## Team
Achilleas Ballanos
Andrii Kernytskyi

Class IT3B Group 8

A custom-built RAG (Retrieval-Augmented Generation) chatbot system that processes and discusses project documents using local LLM integration. The system features both a command-line interface and a modern web UI.

## Features

- Modern Streamlit web interface
- Custom vector database with semantic similarity search
- Local LLM integration (Ollama)
- Document processing (.txt, .docx)
- Dynamic document loading
- Conversation history management
- Automatic context relevance checking
- Command-line interface (alternative)

## Quick Start

1. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Install Ollama:
- Visit https://ollama.com/
- Run: `ollama pull mistral`

3. Start the chatbot:

Web UI (Recommended):
```bash
streamlit run ui.py
```

Command-line interface (Alternative):
```bash
python app.py  # Mac/Linux: python3 app.py
```

## Web Interface

The web interface provides a modern, user-friendly experience:

- Clean chat interface with message bubbles
- Automatic chatbot initialization
- Real-time conversation history
- Database statistics in the sidebar
- Easy chat reset and application shutdown
- Responsive design

### Controls
- Reset Chat: Clear conversation history
- Shutdown: Safely exit the application
- Database Stats: View current document statistics

## Commands

Both interfaces support these commands:
- `/help`  - Show available commands
- `/reset` - Clear conversation history
- `/load`  - Load documents from data/raw
- `/stats` - Show database statistics
- `/exit`  - Exit the chatbot

## Document Management

1. Add documents to `data/raw/`
2. Supported formats: .txt, .docx
3. Documents are automatically processed on startup
4. Documents are automatically deduplicated
5. Vector database persists between sessions

## Technical Details

### Embeddings
- Model: all-MiniLM-L6-v2
- Dimension: 384
- Similarity: Cosine
- Relevance thresholds:
  - Initial search: 0.6
  - Context relevance: 0.7

### Chunking
- Size: 500 characters
- Overlap: 50 characters
- Strategy: Semantic with overlap

### RAG Pipeline
- Custom vector database
- Context window: 5 messages
- Top-k retrieval: 3 chunks
- Local LLM: Ollama (Mistral)
- Automatic context relevance checking

## Project Structure

```
nlp_chatbot_project/
    - data/
        - raw/          # Input documents
        - vector_store/ # Vector database
    - src/
        - preprocessing.py  # Document processing
        - embeddings.py     # Embedding generation
        - vector_db.py      # Vector database
        - chat.py          # Chat interface
    - core/
        - startup.py    # Application initialization
        - ollama.py     # Ollama integration
    - ui.py            # Streamlit web interface
    - app.py           # Command-line interface
    - requirements.txt
```

## Limitations

1. Local processing only
2. Response quality depends on Ollama model
3. Processing time for large documents
4. Memory constraints for large datasets

## Future Work

1. Additional document formats
2. Enhanced conversation persistence
3. Advanced chunking strategies
4. Performance optimizations
5. User authentication
6. Custom model fine-tuning 
