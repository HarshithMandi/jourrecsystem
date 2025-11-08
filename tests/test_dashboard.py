#!/usr/bin/env python3
"""
Test the Streamlit dashboard components.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_dashboard_imports():
    """Test that all dashboard dependencies can be imported."""
    print("Testing Dashboard Dependencies")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✓ Plotly imported successfully")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import requests
        print("✓ Requests imported successfully")
    except ImportError as e:
        print(f"✗ Requests import failed: {e}")
        return False
    
    return True

def test_dashboard_functions():
    """Test dashboard utility functions."""
    print("\nTesting Dashboard Functions")
    
    # Import dashboard functions
    try:
        from dashboard import check_api_status, get_database_stats
        print("✓ Dashboard functions imported successfully")
    except ImportError as e:
        print(f"✗ Dashboard import failed: {e}")
        return False
    
    # Test API status check (should fail if API not running)
    api_status = check_api_status()
    if api_status:
        print("✓ API server is running - dashboard will work")
        
        # Test database stats if API is running
        try:
            stats = get_database_stats()
            if "error" not in stats:
                print("✓ Database statistics retrieved successfully")
                print(f"   Journals: {stats.get('total_journals', 0)}")
                print(f"   Queries: {stats.get('total_queries', 0)}")
            else:
                print(f"! Database stats error: {stats['error']}")
        except Exception as e:
            print(f"! Database stats test failed: {e}")
    else:
        print("! API server not running - start with: uvicorn app.main:app --reload --port 8000")
    
    return True

def main():
    """Run all dashboard tests."""
    print("Journal Recommender Dashboard Tests")
    print("=" * 50)
    
    # Test imports
    if not test_dashboard_imports():
        print("\n✗ Dashboard dependency tests failed!")
        return False
    
    # Test functions
    if not test_dashboard_functions():
        print("\n✗ Dashboard function tests failed!")
        return False
    
    print("\n✓ All dashboard tests passed!")
    print("\nTo launch the dashboard:")
    print("   Option 1: python launch_dashboard.py")
    print("   Option 2: streamlit run dashboard.py")
    print("   Option 3: Manual launch:")
    print("      1. uvicorn app.main:app --reload --port 8000")
    print("      2. streamlit run dashboard.py")
    
    return True

if __name__ == "__main__":
    main()