"""
Main Streamlit App Entry Point for Deployment
This file is required for Streamlit Community Cloud deployment
"""

import sys
from pathlib import Path
import os

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["STREAMLIT_DEPLOYMENT"] = "true"

# Import and run the dashboard
from dashboard import main

if __name__ == "__main__":
    main()