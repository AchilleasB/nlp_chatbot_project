import subprocess
import shutil
import atexit
import signal
import sys
import requests
import time
import threading

ollama_process = None

def cleanup_ollama():
    global ollama_process
    if ollama_process:
        try:
            print("\nStopping Ollama...")
            ollama_process.terminate()
            ollama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Forcing Ollama shutdown...")
            ollama_process.kill()
        finally:
            ollama_process = None

def is_ollama_installed():
    return shutil.which("ollama") is not None

def start_ollama():
    global ollama_process
    try:
        ollama_process = subprocess.Popen(["ollama", "serve"],
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
        time.sleep(2)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_ollama():
    if not is_ollama_installed():
        print("Ollama not installed. Visit https://ollama.com/")
        return False

    try:
        r = requests.get("http://localhost:11434/api/tags")
        if r.status_code == 200:
            return True
    except:
        print("Starting Ollama...")
        if start_ollama():
            try:
                r = requests.get("http://localhost:11434/api/tags")
                if r.status_code == 200:
                    print("Ollama started.")
                    return True
            except:
                pass

    print("Failed to start Ollama.")
    return False

# Register cleanup
atexit.register(cleanup_ollama)

# Only register signals in the main thread
if threading.current_thread() is threading.main_thread():
    try:
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    except ValueError:
        # Ignore signal registration errors in non-main threads
        pass
