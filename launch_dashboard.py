#!/usr/bin/env python3
"""
Launch script for Journal Recommender System
Starts both the API server and Streamlit dashboard.
"""

import subprocess
import sys
import time
import requests
import webbrowser
from pathlib import Path

def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get("http://localhost:8000/ping", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("Journal Recommender System Launcher")
    print("=" * 50)
    
    # Check if API is already running
    if check_api_status():
        print("✓ API server is already running at http://localhost:8000")
        launch_dashboard = True
    else:
        print("Starting API server...")
        
        # Start API server in background
        try:
            api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "app.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--reload"
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
            )
            
            # Wait for API to start
            print("Waiting for API server to start...")
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if check_api_status():
                    print("✓ API server started successfully!")
                    launch_dashboard = True
                    break
                print(f"   Checking... ({i+1}/30)")
            else:
                print("✗ API server failed to start within 30 seconds")
                print("   Please check for errors and try starting manually:")
                print("   uvicorn app.main:app --reload --port 8000")
                launch_dashboard = False
                
        except Exception as e:
            print(f"✗ Failed to start API server: {e}")
            print("   Please start manually: uvicorn app.main:app --reload --port 8000")
            launch_dashboard = False
    
    if launch_dashboard:
        print("\nStarting Streamlit dashboard...")
        print("   Dashboard will open automatically in your browser")
        print("   Or visit: http://localhost:8501")
        
        try:
            # Start Streamlit dashboard
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "dashboard.py",
                "--server.port", "8501",
                "--server.headless", "false"
            ])
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"✗ Failed to start dashboard: {e}")
            print("   Please start manually: streamlit run dashboard.py")
    
    print("\nManual Commands:")
    print("   API Server: uvicorn app.main:app --reload --port 8000")
    print("   Dashboard:  streamlit run dashboard.py")
    print("   Test API:   python test_api.py")

if __name__ == "__main__":
    main()