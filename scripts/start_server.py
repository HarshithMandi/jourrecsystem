#!/usr/bin/env python3
"""
Quick test to start the FastAPI server and verify it's working.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from app.main import app

if __name__ == "__main__":
    print("🚀 Starting Journal Recommender API Server")
    print("📚 Server will be available at: http://localhost:8000")
    print("📖 API Documentation at: http://localhost:8000/docs")
    print("🔍 Health check at: http://localhost:8000/ping")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        log_level="info",
        reload=False
    )
