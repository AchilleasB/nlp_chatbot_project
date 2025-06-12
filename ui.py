import streamlit as st
from pathlib import Path
from core.startup import initialize_chatbot
from core.ollama import check_ollama, cleanup_ollama
import sys
import os

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def shutdown_app():
    """Clean up and exit the application."""
    if st.session_state.chatbot:
        cleanup_ollama()
    # Force exit the Streamlit app
    os._exit(0)

def initialize_chatbot_if_needed():
    """Initialize the chatbot if it hasn't been initialized yet."""
    if not st.session_state.initialized:
        if not check_ollama():
            st.error("Ollama is not installed or running. Please install Ollama from https://ollama.com/")
            return False

        try:
            with st.spinner("Initializing chatbot..."):
                # Fixed model and parameters
                model = "mistral"
                data_dir = Path("data/raw")
                chunk_size = 500
                overlap = 50
                
                chatbot, vector_db = initialize_chatbot(data_dir, model, chunk_size, overlap)
                st.session_state.chatbot = chatbot
                st.session_state.vector_db = vector_db
                st.session_state.messages = []
                st.session_state.initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            return False
    return True

def main():
    st.set_page_config(
        page_title="NLP Chatbot",
        page_icon="🤖",
        layout="centered"
    )

    st.title("NLP Chatbot 🤖")
    
    # Initialize session state
    initialize_session_state()

    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Chat"):
                if st.session_state.chatbot:
                    st.session_state.chatbot.reset_conversation()
                    st.session_state.messages = []
                    st.success("Chat history cleared!")
        
        with col2:
            if st.button("Shutdown", type="primary"):
                st.warning("Shutting down the application...")
                shutdown_app()

        # Display database stats if available
        if st.session_state.vector_db:
            stats = st.session_state.vector_db.get_stats()
            st.header("Database Stats")
            st.write(f"Chunks: {stats['num_chunks']}")
            st.write(f"Total Size: {stats['total_size_mb']:.2f} MB")

    # Initialize chatbot automatically
    if not initialize_chatbot_if_needed():
        st.info("Please make sure Ollama is installed and running.")
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Handle commands
        if prompt.startswith("/"):
            response = st.session_state.chatbot.handle_command(prompt)
            if response == "exit":
                shutdown_app()
        else:
            # Get chatbot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.process_query(prompt)
                st.write(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        cleanup_ollama()
        sys.exit(1) 