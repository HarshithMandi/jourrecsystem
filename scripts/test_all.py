#!/usr/bin/env python3
"""
Comprehensive test script for the Journal Recommender project.
Tests all components and reports any issues.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import traceback
from app.models.base import SessionLocal, engine
from app.models.entities import Journal, JournalProfile, QueryRun, Recommendation
from app.core.config import settings

def test_database_connection():
    """Test database connection and schema"""
    print("Testing database connection...")
    try:
        from sqlalchemy import text
        db = SessionLocal()
        result = db.execute(text("SELECT 1")).fetchone()
        db.close()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_model_imports():
    """Test that all models can be imported"""
    print("Testing model imports...")
    try:
        from app.models.entities import Journal, JournalProfile, Work, QueryRun, Recommendation
        print("✓ All models imported successfully")
        return True
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        from app.core.config import settings
        print(f"✓ Configuration loaded - DB Path: {settings.DB_PATH}")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def test_recommender_service():
    """Test recommender service basic functionality"""
    print("Testing recommender service...")
    try:
        from app.services.recommender import rank_journals
        # Test with empty database (should return empty list)
        result = rank_journals("This is a test abstract about machine learning and artificial intelligence.")
        print(f"✓ Recommender service works - returned {len(result)} results")
        return True
    except Exception as e:
        print(f"✗ Recommender service failed: {e}")
        traceback.print_exc()
        return False

def test_fastapi_app():
    """Test FastAPI application"""
    print("Testing FastAPI application...")
    try:
        from app.main import app
        print("✓ FastAPI application loads successfully")
        return True
    except Exception as e:
        print(f"✗ FastAPI application failed: {e}")
        return False

def test_sample_data_insertion():
    """Test inserting and retrieving sample data"""
    print("Testing sample data insertion...")
    try:
        db = SessionLocal()
        
        # Create a sample journal
        journal = Journal(
            openalex_id="test_journal_123",
            name="Test Journal of Machine Learning",
            display_name="Test Journal of Machine Learning",
            issn="1234-5678",
            is_open_access=True,
            publisher="Test Publisher",
            subjects='[{"id": "ml", "display_name": "Machine Learning"}]'
        )
        db.add(journal)
        db.commit()
        
        # Create a profile for the journal
        profile = JournalProfile(
            journal_id=journal.id,
            scope_text="machine learning artificial intelligence data science",
            tfidf_vector=json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            bert_vector=json.dumps([0.1] * 384)  # Standard BERT dimension
        )
        db.add(profile)
        db.commit()
        
        # Test retrieval
        retrieved = db.query(Journal).filter_by(name="Test Journal of Machine Learning").first()
        if retrieved and retrieved.profile:
            print("✓ Sample data insertion and retrieval successful")
            
            # Clean up
            db.delete(retrieved.profile)
            db.delete(retrieved)
            db.commit()
            
            db.close()
            return True
        else:
            print("✗ Sample data retrieval failed")
            db.close()
            return False
            
    except Exception as e:
        print(f"✗ Sample data insertion failed: {e}")
        traceback.print_exc()
        return False

def test_recommender_with_sample_data():
    """Test recommender with actual sample data"""
    print("Testing recommender with sample data...")
    try:
        db = SessionLocal()
        
        # Create sample journals with profiles
        journals_data = [
            {
                "openalex_id": "test_ml_1",
                "name": "Machine Learning Research",
                "tfidf": [0.8, 0.2, 0.1, 0.0, 0.0],
                "bert": [0.5] * 384
            },
            {
                "openalex_id": "test_bio_1", 
                "name": "Biological Sciences",
                "tfidf": [0.1, 0.0, 0.8, 0.3, 0.1],
                "bert": [0.2] * 384
            }
        ]
        
        created_journals = []
        for data in journals_data:
            journal = Journal(
                openalex_id=data["openalex_id"],
                name=data["name"],
                display_name=data["name"]
            )
            db.add(journal)
            db.commit()
            
            profile = JournalProfile(
                journal_id=journal.id,
                tfidf_vector=json.dumps(data["tfidf"]),
                bert_vector=json.dumps(data["bert"])
            )
            db.add(profile)
            created_journals.append(journal)
        
        db.commit()
        
        # Test recommender
        from app.services.recommender import rank_journals
        results = rank_journals("machine learning and artificial intelligence research")
        
        print(f"✓ Recommender with sample data works - returned {len(results)} results")
        for result in results:
            print(f"  - {result['journal']}: {result['similarity']}")
        
        # Clean up
        for journal in created_journals:
            if journal.profile:
                db.delete(journal.profile)
            db.delete(journal)
        db.commit()
        db.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Recommender with sample data failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("JOURNAL RECOMMENDER - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_config,
        test_database_connection,
        test_model_imports,
        test_fastapi_app,
        test_sample_data_insertion,
        test_recommender_service,
        test_recommender_with_sample_data,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! The project is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
