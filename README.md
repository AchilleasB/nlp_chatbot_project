# NLP Chatbot Project - Local RAG System

## Project Overview

A Python-based chatbot that leverages Retrieval-Augmented Generation (RAG) to provide intelligent, context-aware responses based on your local document collection. The system is designed to run entirely offline, processing documents, generating embeddings, and using a local language model (via Ollama) to answer questions about the document content.

## Features

- **Local Processing**: All document processing, embedding generation, and chat interactions happen on your local machine.
- **RAG System**: Custom implementation of RAG including:
  - Intelligent document chunking with overlap.
  - Local embedding model for converting text to vectors.
  - Vector database for efficient similarity search.
  - Context retrieval to augment language model responses.
- **Document Management**:
  - Support for `.txt` and `.docx` file formats.
  - Easy addition and management of documents via a web UI.
  - Ability to clear or rebuild the knowledge base.
- **Interactive Interfaces**:
  - Command-Line Interface (CLI) for basic interaction and document processing.
  - Streamlit Web Interface for a user-friendly chat and document management experience.

## Installation

### Prerequisites

- Python 3.8 or higher.
- [Ollama](https://ollama.com/) installed and running locally. You will need to pull a language model (e.g., `ollama pull mistral`).
- Git for cloning the repository.

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AchilleasB/nlp_chatbot_project.git
    cd nlp_chatbot_project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # macOS and Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # Windows (PowerShell)
    python -m venv .venv
    .venv\Scripts\Activate.ps1

    # Windows (Command Prompt)
    python -m venv .venv
    .venv\Scripts\activate.bat
    ```

3.  **Install dependencies:**
    ```bash
    # Make sure your virtual environment is activated
    pip install -r requirements.txt
    ```

4.  **Download a language model using Ollama:**
    ```bash
    ollama pull mistral # Or any other preferred model
    ```

## Usage

The application can be run in two modes: Command-Line Interface (CLI) or Streamlit Web Interface.

### Streamlit Web Interface

This is the recommended way to interact with the chatbot and manage documents. It provides a user-friendly chat interface and sidebar controls for document management.

To start the Streamlit app:
```bash
# Make sure your virtual environment is activated
streamlit run app.py
```
Open the provided URL in your web browser.

### Command-Line Interface (CLI)

The CLI allows for basic chat interaction and adding documents directly via the terminal.

To start the CLI chat:
```bash
# Make sure your virtual environment is activated
python app.py
```

To add a document using the CLI:
```bash
# Make sure your virtual environment is activated
python app.py --add-document path/to/your/document.docx
```

## System Design

The system is built around a RAG architecture with custom components for document processing, embedding, and vector storage. It integrates with a local Ollama instance for the language model.

### Component Breakdown:

-   **Document Processing (`src/chunking/`)**: Handles loading, cleaning, and splitting documents into manageable chunks.
-   **Embedding Generation (`src/embeddings/`)**: Implements a local model (e.g., Word2Vec/CBOW) to convert text chunks into numerical vector representations.
-   **Vector Database (`src/vectordb/`)**: Stores the generated vectors and supports efficient similarity search to find relevant document chunks for a given query.
-   **Chat System (`src/chat/`)**: Manages conversation flow, retrieves relevant context from the vector database using RAG, and interacts with the local language model (Ollama) to generate responses.
-   **Core Logic (`src/core.py`)**: Initializes the main application components and provides the bridge between the UI/CLI and the core RAG system.

## Project Structure

```
nlp_chatbot_project/
├── data/
│   ├── raw/              # Original uploaded documents
│   ├── processed/        # (Optional) Intermediate processed document files
│   ├── vector_store/     # Persistent storage for the vector database (vectordb.pkl)
│   └── models/          # Persistent storage for the embedding model (embedding_model.pkl)
├── src/
│   ├── chat/            # Contains ChatManager and related logic
│   ├── chunking/        # Contains TextPreprocessor and chunking logic
│   ├── embeddings/      # Contains embedding model implementation (CBOWModel) and DocumentEmbedder
│   ├── vectordb/        # Contains VectorDB implementation
│   └── core.py          # Entry point for core application logic and CLI
├── requirements.txt     # List of Python dependencies
├── app.py               # Streamlit web application interface and main entry point
└── README.md            # Project description and instructions
```

## Known Limitations

-   Performance for document processing and chat can vary based on system resources and model size.
-   Memory usage scales with the size of the document collection.
-   Currently supports `.txt` and `.docx` file formats.
-   Requires a local Ollama installation and a downloaded model.
-   Limited to the capabilities and language support of the chosen local language model.

---
