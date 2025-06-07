"""Streamlit web interface for the NLP Chatbot application."""

import streamlit as st
from pathlib import Path
import time
from typing import Optional, Tuple, List, Dict
import subprocess
import shutil
import chardet

from src.core import initialize_components, process_document, build_vector_store
from src.utils import ensure_directories_exist, ensure_vector_store_exists
from src.chat import ChatManager
from src.config import DEFAULT_VECTOR_DB_PATH, INPUT_DOCUMENTS_DIR, DEFAULT_EMBEDDING_MODEL_PATH, PROCESSED_DOCUMENTS_DIR

# Set page config with favicon - must be the first Streamlit command
st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_manager' not in st.session_state:
    st.session_state.chat_manager = None
if 'components_initialized' not in st.session_state:
    st.session_state.components_initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'installed_models' not in st.session_state:
    st.session_state.installed_models = []
if 'ollama_status' not in st.session_state:
    st.session_state.ollama_status = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'file_uploader_key_counter' not in st.session_state:
    st.session_state.file_uploader_key_counter = 0
if 'upload_just_finished' not in st.session_state:
    st.session_state.upload_just_finished = False

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

def get_installed_ollama_models() -> Tuple[List[str], bool]:
    """Get list of installed Ollama models from the system.
    
    Returns:
        Tuple of (list of model names, has_models)
    """
    # First check if Ollama is installed and running
    is_installed, status = check_ollama_installation()
    st.session_state.ollama_status = status
    
    if not is_installed:
        st.error(status)
        return [], False
    
    try:
        # Run ollama list command and capture output
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              check=True,
                              timeout=5)
        
        # Parse the output
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # Only header or empty
            st.warning("No models installed yet")
            if st.button("Show Installation Instructions", key="install_instructions_empty_list"):
                st.markdown("""
                To install a model:
                1. Open terminal
                2. Run one of these commands:
                   - `ollama pull mistral` (recommended, ~4GB)
                   - `ollama pull llama2` (~4GB)
                   - `ollama pull codellama` (~4GB)
                3. Wait for the download to complete
                4. Click 'Refresh Models' above
                """)
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
        
        if not model_names:
            st.warning("No models installed yet")
            if st.button("Show Installation Instructions", key="install_instructions_no_models"):
                st.markdown("""
                To install a model:
                1. Open terminal
                2. Run one of these commands:
                   - `ollama pull mistral` (recommended, ~4GB)
                   - `ollama pull llama2` (~4GB)
                   - `ollama pull codellama` (~4GB)
                3. Wait for the download to complete
                4. Click 'Refresh Models' above
                """)
            return [], False
            
        return model_names, True
        
    except subprocess.TimeoutExpired:
        st.error("Timeout while fetching models. Ollama service might be busy or not responding.")
        return [], False
    except subprocess.SubprocessError as e:
        st.error(f"Error communicating with Ollama service: {str(e)}")
        return [], False
    except Exception as e:
        st.error(f"Unexpected error while fetching models: {str(e)}")
        return [], False

def initialize_app(model_name: str) -> Tuple[bool, Optional[str]]:
    """Initialize the application components.
    
    Args:
        model_name: Name of the Ollama model to use
    """
    try:
        # Ensure required directories exist
        ensure_directories_exist()
    
        # Initialize components
        embedding_model, document_embedder, vector_db, chat_manager = initialize_components()
        
        # Update chat manager with selected model
        chat_manager.model_name = model_name
        
        # Store components in session state
        st.session_state.chat_manager = chat_manager
        st.session_state.components_initialized = True
        st.session_state.current_model = model_name
        
        return True, None
    except Exception as e:
        return False, str(e)

def detect_encoding(file_content: bytes) -> str:
    """Detect the encoding of file content.
    
    Args:
        file_content: Raw bytes of the file
        
    Returns:
        Detected encoding
    """
    result = chardet.detect(file_content)
    return result['encoding'] or 'utf-8'

def process_uploaded_document(uploaded_file) -> Tuple[bool, str]:
    """Process an uploaded document and add it to the vector store."""
    try:
        # Read the uploaded file as bytes
        file_content = uploaded_file.getvalue()
        
        # Detect encoding
        encoding = detect_encoding(file_content)
        
        # Try to decode with detected encoding
        try:
            text = file_content.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to latin-1 if detected encoding fails
            text = file_content.decode('latin-1')
        
        # Get components from session state
        embedding_model, document_embedder, vector_db, _ = initialize_components()
        
        # Process the document
        process_document(text, embedding_model, document_embedder, vector_db)
        vector_db.save(DEFAULT_VECTOR_DB_PATH)
        
        return True, f"Document processed and added to database successfully! (Detected encoding: {encoding})"
    except Exception as e:
        return False, f"Error processing document: {str(e)}"

def get_documents_in_raw() -> List[Path]:
    """Get list of documents in the raw documents directory."""
    raw_dir = Path(INPUT_DOCUMENTS_DIR)
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        return []
    
    # Convert generators to lists before combining
    txt_files = list(raw_dir.glob("*.txt"))
    docx_files = list(raw_dir.glob("*.docx"))
    
    # Combine and sort the lists
    all_files = txt_files + docx_files
    return sorted(all_files, key=lambda x: x.name.lower())

def save_uploaded_files() -> Tuple[bool, str]:
    """Save all uploaded files to the raw documents directory."""
    if not st.session_state.uploaded_files:
        return False, "No files to save"
    
    try:
        raw_dir = Path(INPUT_DOCUMENTS_DIR)
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for uploaded_file in st.session_state.uploaded_files:
            if not uploaded_file.name.lower().endswith(('.txt', '.docx')):
                continue
                
            file_path = raw_dir / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            saved_files.append(uploaded_file.name)
        
        st.session_state.uploaded_files = []  # Clear the list after saving
        return True, f"Saved {len(saved_files)} files: {', '.join(saved_files)}"
    except Exception as e:
        return False, f"Error saving files: {str(e)}"

def delete_document(file_path: Path) -> Tuple[bool, str]:
    """Delete a document from the raw documents directory."""
    try:
        file_path.unlink()
        return True, f"Deleted {file_path.name}"
    except Exception as e:
        return False, f"Error deleting {file_path.name}: {str(e)}"

def rebuild_vector_store() -> Tuple[bool, str]:
    """Rebuild the vector store from all documents in the raw directory."""
    try:
        # Get all documents
        documents = get_documents_in_raw()
        if not documents:
            return False, "No documents found to process"
        
        # Initialize components
        embedding_model, document_embedder, vector_db, _ = initialize_components()
        
        # Process each document
        for doc_path in documents:
            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()
            process_document(text, embedding_model, document_embedder, vector_db)
        
        # Save the vector store
        vector_db.save(DEFAULT_VECTOR_DB_PATH)
        return True, f"Successfully rebuilt vector store with {len(documents)} documents"
    except Exception as e:
        return False, f"Error rebuilding vector store: {str(e)}"

def clear_knowledge_base():
    """Clear all documents and the vector store."""
    try:
        # Clear raw documents directory
        raw_dir = Path(INPUT_DOCUMENTS_DIR)
        if raw_dir.exists():
            for file in raw_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        
        # Clear processed documents directory
        processed_dir = Path(PROCESSED_DOCUMENTS_DIR)
        if processed_dir.exists():
            for file in processed_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        
        # Clear vector store
        vector_db_path = Path(DEFAULT_VECTOR_DB_PATH)
        if vector_db_path.exists():
            vector_db_path.unlink()
        
        # Reset session state
        st.session_state.messages = []
        st.session_state.components_initialized = False
        st.session_state.chat_manager = None
        
        return True, "Knowledge base cleared successfully"
    except Exception as e:
        return False, f"Error clearing knowledge base: {str(e)}"

def main():
    """Main Streamlit application entry point."""
    st.title("🤖 NLP Chatbot")
    
    # Sidebar for model selection and document management
    with st.sidebar:
        st.header("Model Selection")
        
        # Check Ollama status and get installed models
        models, has_models = get_installed_ollama_models()
        
        if has_models:
            # Model selection dropdown
            selected_model = st.selectbox(
                "Select Model",
                models,
                index=models.index(st.session_state.current_model) if st.session_state.current_model in models else 0
            )
            
            # Initialize button
            if st.button("Initialize Model", disabled=st.session_state.components_initialized and st.session_state.current_model == selected_model):
                success, error = initialize_app(selected_model)
                if success:
                    st.success("Model initialized successfully!")
                    st.rerun()
                else:
                    st.error(f"Error initializing model: {error}")
        
        st.header("Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['txt', 'docx'],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.file_uploader_key_counter}"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files.extend(uploaded_files)
            st.session_state.upload_just_finished = True
        
        # Process uploaded files button
        if st.session_state.uploaded_files:
            if st.button("Process Uploaded Files"):
                with st.spinner("Processing files..."):
                    success, message = save_uploaded_files()
                    if success:
                        st.success(message)
                        st.session_state.file_uploader_key_counter += 1
                        st.rerun()
                    else:
                        st.error(message)
        
        # Show existing documents
        st.subheader("Existing Documents")
        documents = get_documents_in_raw()
        
        if not documents:
            st.info("No documents uploaded yet")
        else:
            for doc_path in documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(doc_path.name)
                with col2:
                    if st.button("🗑️", key=f"delete_{doc_path.name}"):
                        success, message = delete_document(doc_path)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        
        # Rebuild vector store button
        if documents:
            if st.button("Rebuild Vector Store"):
                with st.spinner("Rebuilding vector store..."):
                    success, message = rebuild_vector_store()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # Clear knowledge base button
        if st.button("Clear Knowledge Base", type="secondary"):
            if st.checkbox("I understand this will delete all documents and the vector store"):
                success, message = clear_knowledge_base()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    # Main chat interface
    if not st.session_state.components_initialized:
        st.info("Please select a model and initialize it in the sidebar")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from chat manager
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_manager.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 