"""Main entry point for the NLP Chatbot application with Streamlit web interface."""

import streamlit as st
from pathlib import Path
import time
from typing import Optional, Tuple, List, Dict
import subprocess
import shutil
import chardet

from src.core import initialize_components, process_document, build_vector_store, run_cli
from src.utils import ensure_directories_exist, ensure_vector_store_exists, print_inspection_results
from src.chat import ChatManager
from src.config import DEFAULT_VECTOR_DB_PATH, INPUT_DOCUMENTS_DIR, DEFAULT_EMBEDDING_MODEL_PATH, PROCESSED_DOCUMENTS_DIR

# Constants
DEFAULT_VECTOR_DB_PATH = DEFAULT_VECTOR_DB_PATH  # Use the path from config

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
        
        if not saved_files:
            return False, "No valid files to save"
            
        # Clear the uploaded files after successful save
        st.session_state.uploaded_files = []
        return True, f"Successfully saved: {', '.join(saved_files)}"
    except Exception as e:
        return False, f"Error saving files: {str(e)}"

def delete_document(file_path: Path) -> Tuple[bool, str]:
    """Delete a document from the raw documents directory."""
    try:
        file_path.unlink()
        return True, f"Document {file_path.name} deleted successfully"
    except Exception as e:
        return False, f"Error deleting document: {str(e)}"

def rebuild_vector_store() -> Tuple[bool, str]:
    """Rebuild the vector store from documents in data/raw."""
    try:
        # Ensure directories exist before rebuilding
        ensure_directories_exist()

        embedding_model, document_embedder, vector_db, _ = initialize_components()
        build_vector_store(
            input_dir_path=INPUT_DOCUMENTS_DIR,
            vector_db_path=DEFAULT_VECTOR_DB_PATH,
            embedding_model_path=DEFAULT_EMBEDDING_MODEL_PATH,
            embedding_model=embedding_model,
            document_embedder=document_embedder,
            vector_db=vector_db
        )
        # Do NOT set components_initialized to False here; initialize_app will handle it on rerun
        return True, "Knowledge base rebuilt successfully"
    except Exception as e:
        return False, f"Error rebuilding knowledge base: {str(e)}"

def clear_knowledge_base():
    """Deletes all uploaded files, processed files, and the vector database."""
    import shutil
    import os
    from pathlib import Path
    import streamlit as st

    try:
        # Delete raw documents directory
        raw_dir = Path(INPUT_DOCUMENTS_DIR)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            print(f"Cleared raw documents directory: {raw_dir}") # Optional: log to console

        # Delete processed documents directory
        processed_dir = Path(PROCESSED_DOCUMENTS_DIR)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
            print(f"Cleared processed documents directory: {processed_dir}") # Optional: log to console

        # Delete vector database file
        vector_db_file = Path(DEFAULT_VECTOR_DB_PATH)
        if vector_db_file.exists():
            vector_db_file.unlink()
            print(f"Deleted vector database file: {vector_db_file}") # Optional: log to console

        # Reset relevant session state variables
        st.session_state.uploaded_files = []
        st.session_state.last_uploaded_files_hash = None
        st.session_state.components_initialized = False # Force re-initialization
        # Optionally, clear chat history if desired, but not explicitly requested:
        # st.session_state.messages = []

        return True, "Knowledge base cleared successfully!"
    except Exception as e:
        return False, f"Error clearing knowledge base: {str(e)}"

def main():
    """Main Streamlit application."""
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 NLP Chatbot")
        
        # Document Management Section
        st.header("📚 Document Management")
        
        # Document Upload Section
        st.subheader("Add Documents")
        
        # Use a dynamic key for the file uploader to force reset after upload
        # Initialize file_uploader_key_counter in session state if not present
        if 'file_uploader_key_counter' not in st.session_state:
            st.session_state.file_uploader_key_counter = 0

        uploaded_files_uploader = st.file_uploader(
            "Choose files",
            type=['txt', 'docx'],
            accept_multiple_files=True,
            help="Upload one or more documents to add to the chatbot's knowledge base",
            key=f"file_uploader_key_{st.session_state.file_uploader_key_counter}"
        )

        # Update session state.uploaded_files based on the current state of the uploader widget
        # This list is used by the save_uploaded_files function.
        if uploaded_files_uploader is not None and len(uploaded_files_uploader) > 0:
            # Update session state with the files from the uploader
            st.session_state.uploaded_files = uploaded_files_uploader
        elif st.session_state.get('uploaded_files') and (uploaded_files_uploader is None or len(uploaded_files_uploader) == 0):
             # If the uploader is empty, clear the session state list as well
             st.session_state.uploaded_files = []

        # Only show selected files count and upload button if there are pending uploads shown by the uploader widget
        # We now directly check the value of uploaded_files_uploader from the current rerun
        if uploaded_files_uploader is not None and len(uploaded_files_uploader) > 0:
            st.info(f"{len(uploaded_files_uploader)} file(s) selected for upload") # Use uploader variable directly

            # Upload button - only show if files are selected in the uploader
            # The save_uploaded_files function reads from st.session_state.uploaded_files
            if st.button("📥 Upload Selected Files", type="primary", key="upload_button"):
                with st.spinner("Uploading files..."):
                    success, message = save_uploaded_files()
                    if success:
                        st.toast("📄 Files uploaded successfully!", icon="📥")
                        # Clear all upload-related session state and increment counter for reset
                        st.session_state.uploaded_files = [] # Explicitly clear session state list
                        st.session_state.file_uploader_key_counter += 1
                        # Use rerun to refresh the page state and reset the uploader widget display
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)

        # Current Documents Section
        st.subheader("Current Documents")
        documents = get_documents_in_raw()
        
        if documents:
            # Create a container for the document list
            doc_container = st.container()
            with doc_container:
                for doc in documents:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(doc.name)
                    with col2:
                        if st.button("🗑️", key=f"delete_{doc.name}", help=f"Delete {doc.name}"):
                            success, message = delete_document(doc)
                            if success:
                                # Show toast notification at the top of the screen
                                st.toast(f"📄 {doc.name} has been removed from the knowledge base", icon="🗑️")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)
            
            # Rebuild button
            if st.button("🔄 Rebuild Knowledge Base", type="secondary"):
                with st.spinner("Rebuilding knowledge base..."):
                    success, message = rebuild_vector_store()
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)

        else:
            st.info("No documents in the knowledge base yet")

        # Clear Knowledge Base button - Moved outside the `if documents:` block
        if st.button("❌ Clear Knowledge Base", type="secondary"):
            with st.spinner("Clearing knowledge base..."):
                success, message = clear_knowledge_base()
                if success:
                    st.toast("🗑️ Knowledge base cleared!", icon="✅")
                    time.sleep(1) # Give toast time to show
                    st.rerun()
                else:
                    st.error(message)
        
        st.divider()
        
        # Ollama Status Section
        st.header("🤖 Ollama Status")
        is_installed, status = check_ollama_installation()
        
        if is_installed:
            st.success(status)
        else:
            st.error(status)
            with st.expander("How to fix this"):
                st.markdown("""
                1. Install Ollama from https://ollama.ai
                2. Start the Ollama service:
                   - On macOS/Linux: Run `ollama serve` in terminal
                   - On Windows: Start Ollama from the Start menu
                3. Refresh this page
                """)
        
        # Model Selection Section
        if is_installed:
            st.header("Model Selection")
            
            # Get installed models if not already cached
            if not st.session_state.installed_models:
                with st.spinner("Fetching installed models..."):
                    st.session_state.installed_models, has_models = get_installed_ollama_models()
                    st.session_state.has_models = has_models
            
            if st.session_state.installed_models:
                # Select box
                selected_model = st.selectbox(
                    "Select Model",
                    st.session_state.installed_models,
                    index=st.session_state.installed_models.index(st.session_state.current_model) 
                        if st.session_state.current_model in st.session_state.installed_models 
                        else 0,
                    help="Select an installed Ollama model to use for chat"
                )
                
                # Handle model change
                if selected_model != st.session_state.current_model:
                    st.session_state.current_model = selected_model
                    with st.spinner(f"Switching to model: {selected_model}..."):
                        success, error = initialize_app(selected_model)
                        if not success:
                            st.error(f"Failed to switch model: {error}")
                            st.stop()
                        st.session_state.messages = []
                        st.rerun()
                
                # Model management buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh Models"):
                        with st.spinner("Refreshing model list..."):
                            st.session_state.installed_models, has_models = get_installed_ollama_models()
                            st.session_state.has_models = has_models
                            if not has_models:
                                st.session_state.current_model = None
                                st.session_state.components_initialized = False
                            st.rerun()
                with col2:
                    if st.button("📖 Install Model"):
                        with st.expander("Installation Instructions"):
                            st.markdown("""
                            To install a new model:
                            1. Open terminal
                            2. Run: `ollama pull <model_name>`
                            3. Common models: mistral, llama2, codellama
                            4. Click 'Refresh Models' after installation
                            """)
                
                # Current model info box
                st.info(f"Current Model: {st.session_state.current_model}")
            else:
                st.warning("No models available. Please install a model first.")
                st.session_state.has_models = False
                st.session_state.current_model = None
                st.session_state.components_initialized = False
        
        st.divider()
        
        # Chat Management Section
        st.header("Chat Management")
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.chat_manager:
                st.session_state.chat_manager.clear_history()
            st.session_state.messages = []
            st.rerun()
        
        # System Status Section
        st.header("System Status")
        if st.session_state.components_initialized:
            st.success("✅ System Ready")
            if st.session_state.chat_manager:
                summary = st.session_state.chat_manager.get_conversation_summary()
                st.info(f"Messages in current session: {summary['message_count']}")
        else:
            st.warning("⚠️ Initializing...")

    # Main chat interface
    st.title("Chat with AI Assistant")
    
    # Check if we have models available
    if not st.session_state.get('has_models', False) or not st.session_state.current_model:
        st.warning("No models available for chat")
        if st.button("Show Installation Instructions", key="install_instructions_main"):
            st.markdown("""
            To install a model:
            1. Open terminal
            2. Run one of these commands:
               - `ollama pull mistral` (recommended, ~4GB)
               - `ollama pull llama2` (~4GB)
               - `ollama pull codellama` (~4GB)
            3. Wait for the download to complete
            4. Click 'Refresh Models' in the sidebar
            """)
        st.stop()
    
    # Initialize components if not already done
    if not st.session_state.components_initialized:
        with st.spinner("Initializing system components..."):
            success, error = initialize_app(st.session_state.current_model)
            if not success:
                st.error(f"Failed to initialize: {error}")
                st.stop()
            else:
                # Add an extra rerun to ensure status update is reflected
                st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("context_used"):
                with st.expander("Context Used"):
                    st.markdown(message["context_used"][0])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner(f"Thinking using {st.session_state.current_model}..."):
                try:
                    response = st.session_state.chat_manager.chat(prompt)
                    message_placeholder.markdown(response)
                    
                    # Get context if used
                    context = None
                    if st.session_state.chat_manager.conversation_history:
                        last_message = st.session_state.chat_manager.conversation_history[-1]
                        if last_message.context_used:
                            context = last_message.context_used[0]
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "context_used": [context] if context else None
                    })
                    
                    # Show context if available
                    if context:
                        with st.expander("Context Used"):
                            st.markdown(context)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    message_placeholder.markdown("I apologize, but I encountered an error. Please try again.")

# Check if the script is being run by Streamlit
def is_running_with_streamlit():
    try:
        # This check is a common way to detect if Streamlit is the runner
        # Accessing a Streamlit internal variable or function that\'s only available when running with streamlit
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except (ImportError, RuntimeError):
        # If streamlit.runtime or get_script_run_ctx is not available,
        # or if get_script_run_ctx raises a RuntimeError, it's not running with streamlit
        return False
    except Exception:
        # Catch any other unexpected exceptions
        return False

if __name__ == "__main__":
    if is_running_with_streamlit():
        # Run the Streamlit app
        main()
    else:
        # Run the CLI app
        run_cli()
        
