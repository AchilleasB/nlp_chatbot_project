# NLP Chatbot with RAG

A Python-based chatbot that uses Retrieval-Augmented Generation (RAG) to provide context-aware responses based on a local document collection. The system processes documents, creates embeddings, and uses them to enhance the chat responses with relevant information.

## Features

- **Local Processing**: All processing is done locally using Ollama for the language model

- **RAG Implementation**: Custom-built RAG system with:
  - Document chunking with overlap
  - Local embedding generation
  - Vector similarity search
  - Context-aware responses

- **Document Management**:
  - Support for .txt and .docx files
  - Automatic document processing and chunking
  - Vector store for efficient retrieval

- **Interactive Chat**:
  - Context-aware responses using top 3 most relevant chunks
  - Conversation history management
  - Detailed logging of context selection

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed and running locally
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd npl_chatbot_project
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Ollama and pull the required model:
```bash
ollama pull mistral
```

## Usage
### Interactive Chat

Start the interactive chat interface in the CLI:

```bash
python app.py
```

### Document Processing

You can add new documents to the system using the `--add-document` flag:

```bash
python app.py --add-document path/to/your/document.docx
```

This will:
1. Process the document
2. Split it into chunks with overlap
3. Generate embeddings
4. Add the vectors to the database

### Chat Features

- The system retrieves the top 3 most relevant chunks for each query
- Responses are enhanced with context from these chunks
- Conversation history is maintained for context
- Detailed logging shows:
  - Relevance scores for retrieved chunks
  - Context selection process
  - Model response generation

## Project Structure

```
npl_chatbot_project/
├── data/
│   ├── raw/              # Input documents
│   ├── processed/        # Processed chunks
│   ├── vector_store/     # Vector database
│   └── models/          # Embedding models
├── src/
│   ├── chat/            # Chat functionality
│   ├── chunking/        # Document processing
│   ├── embeddings/      # Embedding generation
│   ├── vectordb/        # Vector database
│   └── core.py          # Main application
├── requirements.txt
└── README.md
```

## Notes

- The system uses a local embedding model for document processing
- All processing is done offline for privacy and performance
- The vector store is automatically created and updated as documents are added
- Conversation history is maintained in memory and cleared when the application exits
