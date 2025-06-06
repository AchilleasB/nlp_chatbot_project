# NLP Project Chatbot

Submission Date: 2024-06-13
Team Members:
- Achilleas Ballanos
- Andrii Kernitskii

## Project Overview

A Python-based chatbot that uses Retrieval-Augmented Generation (RAG) to provide context-aware responses based on a local document collection. The system processes documents, creates embeddings, and uses them to enhance the chat responses with relevant information.

### Features

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

## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed and running locally
- Required Python packages (see `requirements.txt`)

### Setup Steps

1. Clone the repository:
```bash
# Unix/MacOS
git clone <repository-url>
cd npl_chatbot_project

# Windows (PowerShell)
git clone <repository-url>
cd npl_chatbot_project
```

2. Create and activate a virtual environment:
```bash
# Unix/MacOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat
```

3. Install dependencies:
```bash
# Unix/MacOS
pip install -r requirements.txt

# Windows
python -m pip install -r requirements.txt
```

4. Start Ollama and pull the required model:
```bash
# Unix/MacOS
ollama pull mistral

# Windows (PowerShell/Command Prompt)
ollama pull mistral
```

## Usage Guide

### Interactive Chat

Start the interactive chat interface:
```bash
# Unix/MacOS
python3 app.py

# Windows
python app.py
```

### Document Processing

Add new documents to the system:
```bash
# Unix/MacOS
python3 app.py --add-document path/to/your/document.docx

# Windows
python app.py --add-document path\to\your\document.docx
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

## Architecture Description

### Project Structure
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

### Component Overview

1. **Document Processing** (`src/chunking/`):
   - Text splitting and chunking
   - Document metadata extraction
   - Section detection and organization

2. **Embedding Generation** (`src/embeddings/`):
   - Custom Word2Vec implementation
   - Document-level embedding creation
   - Vector similarity calculations

3. **Vector Database** (`src/vectordb/`):
   - Efficient vector storage and retrieval
   - Similarity search implementation
   - Persistent storage management

4. **Chat System** (`src/chat/`):
   - RAG-based response generation
   - Context management
   - Conversation history handling

## NLP Approach Explanation

### Document Processing
- **Chunking Strategy**: Documents are split into semantically meaningful chunks with overlap
- **Section Detection**: Uses pattern matching to identify document sections and headers
- **Metadata Extraction**: Preserves document structure and context

### Embedding Generation
- **Word2Vec Implementation**: Custom CBOW model for word embeddings
- **Document Embedding**: Combines word vectors to create document-level representations
- **Training Process**: Uses negative sampling for efficient training

### RAG Implementation
- **Context Retrieval**: Uses vector similarity to find relevant document chunks
- **Response Generation**: Combines retrieved context with LLM responses
- **Relevance Scoring**: Implements cosine similarity for chunk ranking

### Key NLP Techniques
1. **Text Preprocessing**:
   - Sentence splitting
   - Section detection
   - Metadata handling

2. **Embedding Generation**:
   - Word-level embeddings
   - Document-level embeddings
   - Vector similarity calculations

3. **Context Management**:
   - Chunk relevance scoring
   - Context selection
   - Response enhancement

## Known Limitations

1. **Processing Limitations**:
   - Large documents may take longer to process
   - Memory usage increases with document size
   - Limited to .txt and .docx file formats

2. **Model Limitations**:
   - Requires local Ollama installation
   - Dependent on model quality and size
   - Limited to English language support

3. **Performance Considerations**:
   - Initial document processing can be slow
   - Vector search performance depends on database size
   - Memory usage scales with document collection size

4. **Technical Constraints**:
   - All processing must be done locally
   - No external API calls allowed
   - Limited to offline models via Ollama
