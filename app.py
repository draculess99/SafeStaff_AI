import os
import sys
import time
import subprocess
import signal
from backend.model import train_model, CSV_PATH, MODEL_PATH

def main():
    print("=========================================================")
    print("   SafeStaff AI - 2-Tier Multi-Agent Staffing System     ")
    print("=========================================================")

    # 1. Train XGBoost Model on Startup if not trained
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Auto-generating data and training XGBoost...")
        train_model(CSV_PATH, MODEL_PATH)
    else:
        print("XGBoost model found. Skipping training.")

    # 2. Launch Flask Backend (Port 5000)
    print("\nStarting Flask API Backend on http://127.0.0.1:5000...")
    backend_env = os.environ.copy()
    backend_env["PORT"] = "5000"  # Force internal port for Flask
    
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "backend.server"],
        env=backend_env
    )

    # 3. Wait for Flask to boot up
    time.sleep(2)

    # 4. Launch Streamlit Frontend Dashboard (Port $PORT or 8501)
    port = os.environ.get("PORT", "8501")
    print(f"Starting Streamlit Dashboard on http://0.0.0.0:{port}...")
    
    # Force Streamlit to use the locally running backend
    env = os.environ.copy()
    env["API_BASE_URL"] = "http://127.0.0.1:5000"
    env["BACKEND_URL"] = "http://127.0.0.1:5000"
    
    frontend_script = os.path.join("frontend", "dashboard.py")
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", frontend_script, "--server.port", port, "--server.address", "0.0.0.0"],
        env=env
    )

    print("\nSafeStaff AI is running successfully!")
    print(f"To open the UI, navigate to http://localhost:{port} in your browser.")
    print("Press Ctrl+C to terminate both servers.")
    print("=========================================================\n")

    # Keep script running and catch Ctrl+C to stop subprocesses
    try:
        while True:
            # Check if backend or frontend died
            if backend_proc.poll() is not None:
                print("Error: Flask Backend stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("Error: Streamlit Frontend stopped unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown signal received. Stopping servers...")
    finally:
        # Terminate processes
        backend_proc.terminate()
        frontend_proc.terminate()
        
        # Wait for shutdown
        backend_proc.wait()
        frontend_proc.wait()
        print("Both servers shut down cleanly. Goodbye!")

if __name__ == "__main__":
    main()
