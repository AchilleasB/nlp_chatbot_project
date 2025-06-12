# NLP Project Chatbot

A custom-built RAG (Retrieval-Augmented Generation) chatbot system that processes and discusses project documents using local LLM integration.

## Features

- Custom vector database with similarity search
- Local LLM integration (Ollama)
- Document processing (.txt, .docx)
- Dynamic document loading
- Conversation history management
- Simple command-line interface

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

3. Start the chatbot (Windows):
```bash
python app.py # Mac/Linux: python3 app.py
```

## Commands

- `/help`  - Show available commands
- `/reset` - Clear conversation history
- `/load`  - Load documents from data/raw
- `/stats` - Show database statistics
- `/exit`  - Exit the chatbot

## Document Management

1. Add documents to `data/raw/`
2. Supported formats: .txt, .docx
3. Use `/load` to process documents
4. Documents are automatically deduplicated

## Technical Details

### Embeddings
- Model: all-MiniLM-L6-v2
- Dimension: 384
- Similarity: Cosine

### Chunking
- Size: 500 characters (configurable)
- Overlap: 50 characters (configurable)
- Strategy: Semantic with overlap

### RAG Pipeline
- Custom vector database
- Context window: 5 messages
- Top-k retrieval: 3 chunks
- Local LLM: Ollama (Mistral)

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
    - app.py
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
3. Advanced chunking
4. Performance optimizations 
