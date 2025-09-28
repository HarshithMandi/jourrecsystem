#!/usr/bin/env python3
"""
Complete Journal Recommender API Launcher
==========================================

This script sets up and runs the entire journal recommender system as an API:
1. ✅ Environment check and package installation
2. 🗃️ Database initialization  
3. 📡 Data ingestion from OpenAlex API
4. 🧠 ML vector building (TF-IDF + BERT)
5. 🧪 System testing
6. 🚀 API server startup with live testing

Run this script to get the complete API system running with one command!
"""

import sys
import os
import subprocess
import time
import threading
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def print_section(title, emoji="🔧"):
    """Print a formatted section header"""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))

def run_command(cmd, description, check_output=False):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        if check_output:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def check_database_exists():
    """Check if database file exists"""
    db_path = project_root / "journal_recommender.db"
    return db_path.exists()

def check_has_data():
    """Quick check if database has data"""
    try:
        from app.models.base import SessionLocal
        from app.models.entities import Journal
        
        db = SessionLocal()
        count = db.query(Journal).count()
        db.close()
        return count > 0
    except Exception:
        return False

def main():
    """Main execution function"""
    print("🚀 Journal Recommender API - Complete Setup & Launch")
    print("====================================================")
    print(f"📁 Working directory: {project_root}")
    print(f"🐍 Python version: {sys.version}")
    
    # Check virtual environment
    if check_virtual_environment():
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected - consider using venv")
    
    # Step 1: Environment Check & Package Installation
    print_section("Step 1: Environment Setup", "🔧")
    
    print("📦 Installing required packages...")
    packages_to_install = [
        "fastapi", "uvicorn[standard]", "pydantic", 
        "sqlalchemy", "requests", "scikit-learn", 
        "sentence-transformers", "numpy", "tqdm"
    ]
    
    for package in packages_to_install:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Step 2: Database Setup (if needed)
    if not check_database_exists():
        print_section("Step 2: Database Initialization", "🗃️")
        
        if not run_command("python scripts/init_db.py", "Database initialization"):
            print("❌ Database initialization failed - cannot continue")
            return False
    else:
        print_section("Step 2: Database Check", "🗃️")
        print("✅ Database file exists")
    
    # Step 3: Data Ingestion (if needed)
    if not check_has_data():
        print_section("Step 3: Data Ingestion from OpenAlex", "📡")
        
        if not run_command("python scripts/ingest_openalex.py", "Data ingestion"):
            print("❌ Data ingestion failed - cannot continue") 
            return False
    else:
        print_section("Step 3: Data Check", "📡")
        print("✅ Database contains journal data")
    
    # Step 4: ML Vector Building (check if needed)
    print_section("Step 4: Machine Learning Vector Building", "🧠")
    
    try:
        from app.models.base import SessionLocal
        from app.models.entities import JournalProfile
        
        db = SessionLocal()
        profiles_with_vectors = db.query(JournalProfile).filter(
            JournalProfile.tfidf_vector.isnot(None),
            JournalProfile.bert_vector.isnot(None)
        ).count()
        db.close()
        
        if profiles_with_vectors == 0:
            if not run_command("python scripts/build_vectors.py", "Vector building"):
                print("❌ Vector building failed - cannot continue")
                return False
        else:
            print(f"✅ Found {profiles_with_vectors} journals with ML vectors")
    except Exception:
        # If we can't check, just try to build vectors
        if not run_command("python scripts/build_vectors.py", "Vector building"):
            print("❌ Vector building failed - cannot continue")
            return False
    
    # Step 5: Quick System Test
    print_section("Step 5: System Testing", "🧪")
    
    print("🔍 Testing recommendation system...")
    test_query = """
    This research investigates machine learning algorithms for protein structure prediction
    using deep neural networks and evolutionary information to improve accuracy.
    """
    
    try:
        from app.services.recommender import rank_journals
        results = rank_journals(test_query, top_k=3)
        if results:
            print("✅ Recommendation system working!")
            print("Top 3 recommendations:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result['journal']} (score: {result['similarity']:.3f})")
        else:
            print("⚠️  No recommendations returned")
    except Exception as e:
        print(f"⚠️  Recommendation test failed: {e}")
    
    # Step 6: Start API Server
    print_section("Step 6: Starting API Server", "🚀")
    
    print("🌐 Starting FastAPI server...")
    print("📋 Server will be available at:")
    print("   • API Endpoints: http://localhost:8000/api/")  
    print("   • Interactive Docs: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/ping")
    print("\n🔧 Starting server...")
    
    # Start server in background
    try:
        import uvicorn
        from app.main import app
        
        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        print("✅ API server started in background")
        time.sleep(3)  # Give server time to start
        
        # Step 7: Test API
        print_section("Step 7: API Testing", "📊")
        
        print("🧪 Running API tests...")
        if run_command("python test_api.py", "API testing"):
            print("\n🎉 SUCCESS! Complete API system is running!")
            print("\n🌐 Your Journal Recommender API is ready!")
            print("   📡 Main endpoint: http://localhost:8000/api/recommend")
            print("   📚 Documentation: http://localhost:8000/docs")
            print("   🔍 Health check: http://localhost:8000/ping")
            
            # Sample usage examples
            print("\n💡 Example API usage:")
            print("   🔗 Web Interface: http://localhost:8000/docs")
            print("\n   🖥️  curl command:")
            print('''   curl -X POST "http://localhost:8000/api/recommend" \\
        -H "Content-Type: application/json" \\
        -d '{
          "abstract": "Machine learning for protein structure prediction using neural networks",
          "top_k": 5
        }' ''')
            
            print("\n   🐍 Python example:")
            print('''   import requests
   response = requests.post("http://localhost:8000/api/recommend", json={
       "abstract": "Your research abstract here",
       "top_k": 10
   })
   print(response.json())''')
            
            print("\n⏸️  Server is running - Press Ctrl+C to stop")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Shutting down server...")
                return True
        else:
            print("⚠️  API tests failed, but server is still running")
            print("   Check http://localhost:8000/docs to verify manually")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Shutting down server...")
                return True
            
    except ImportError as e:
        print(f"❌ Could not start API server: {e}")
        print("Try installing missing packages: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        print("\nYou can manually start the server with:")
        print("python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Launch interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)