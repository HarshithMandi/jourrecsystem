#!/usr/bin/env python3
"""
Database System Test Script
Tests all database operations and data integrity.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import traceback
from datetime import datetime
from sqlalchemy import text, select
from app.models.base import SessionLocal, engine, Base
from app.models.entities import Journal, JournalProfile, Work, QueryRun, Recommendation
from app.core.config import settings

def test_database_connection():
    """Test basic database connectivity"""
    print("🔍 Testing database connection...")
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT 1 as test")).fetchone()
        db.close()
        print(f"✅ Database connection successful - Result: {result[0]}")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_database_schema():
    """Test database schema and tables"""
    print("\n🔍 Testing database schema...")
    try:
        db = SessionLocal()
        
        # Check if all tables exist
        tables_query = text("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name;
        """)
        
        tables = db.execute(tables_query).fetchall()
        table_names = [table[0] for table in tables]
        
        expected_tables = ['journals', 'journal_profiles', 'works', 'query_runs', 'recommendations']
        
        print(f"📋 Found tables: {table_names}")
        
        for expected in expected_tables:
            if expected in table_names:
                print(f"✅ Table '{expected}' exists")
            else:
                print(f"❌ Table '{expected}' missing")
                db.close()
                return False
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def test_journal_operations():
    """Test Journal model CRUD operations"""
    print("\n🔍 Testing Journal CRUD operations...")
    try:
        db = SessionLocal()
        
        # Create
        test_journal = Journal(
            openalex_id="test_journal_db_001",
            name="Test Database Journal",
            display_name="Test Database Journal Display",
            issn="9999-8888",
            eissn="9999-8889",
            is_open_access=True,
            publisher="Test Database Publisher",
            impact_factor=2.5,
            subjects='[{"id": "cs", "display_name": "Computer Science"}]'
        )
        
        db.add(test_journal)
        db.commit()
        print("✅ Journal creation successful")
        
        # Read
        retrieved = db.query(Journal).filter_by(openalex_id="test_journal_db_001").first()
        if retrieved and retrieved.name == "Test Database Journal":
            print("✅ Journal retrieval successful")
        else:
            print("❌ Journal retrieval failed")
            return False
        
        # Update
        retrieved.publisher = "Updated Test Publisher"
        db.commit()
        
        updated = db.query(Journal).filter_by(openalex_id="test_journal_db_001").first()
        if updated.publisher == "Updated Test Publisher":
            print("✅ Journal update successful")
        else:
            print("❌ Journal update failed")
            return False
        
        # Delete
        db.delete(updated)
        db.commit()
        
        deleted_check = db.query(Journal).filter_by(openalex_id="test_journal_db_001").first()
        if deleted_check is None:
            print("✅ Journal deletion successful")
        else:
            print("❌ Journal deletion failed")
            return False
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Journal operations failed: {e}")
        traceback.print_exc()
        return False

def test_journal_profile_operations():
    """Test JournalProfile model operations"""
    print("\n🔍 Testing JournalProfile operations...")
    try:
        db = SessionLocal()
        
        # Create a journal first
        journal = Journal(
            openalex_id="test_profile_journal_001",
            name="Profile Test Journal",
            display_name="Profile Test Journal"
        )
        db.add(journal)
        db.commit()
        
        # Create profile
        profile = JournalProfile(
            journal_id=journal.id,
            scope_text="machine learning artificial intelligence data science",
            tfidf_vector=json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            bert_vector=json.dumps([0.01] * 384),  # BERT dimension
            total_articles=150
        )
        
        db.add(profile)
        db.commit()
        print("✅ JournalProfile creation successful")
        
        # Test relationship
        retrieved_journal = db.query(Journal).filter_by(id=journal.id).first()
        if retrieved_journal.profile and retrieved_journal.profile.total_articles == 150:
            print("✅ Journal-Profile relationship successful")
        else:
            print("❌ Journal-Profile relationship failed")
            return False
        
        # Test vector retrieval and parsing
        tfidf_data = json.loads(profile.tfidf_vector)
        bert_data = json.loads(profile.bert_vector)
        
        if len(tfidf_data) == 5 and len(bert_data) == 384:
            print("✅ Vector data storage and retrieval successful")
        else:
            print("❌ Vector data storage failed")
            return False
        
        # Cleanup
        db.delete(profile)
        db.delete(journal)
        db.commit()
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ JournalProfile operations failed: {e}")
        traceback.print_exc()
        return False

def test_work_operations():
    """Test Work model operations"""
    print("\n🔍 Testing Work operations...")
    try:
        db = SessionLocal()
        
        # Create a journal first
        journal = Journal(
            openalex_id="test_work_journal_001",
            name="Work Test Journal"
        )
        db.add(journal)
        db.commit()
        
        # Create work
        work = Work(
            openalex_id="test_work_001",
            title="Test Research Paper on Machine Learning",
            abstract="This paper explores advanced machine learning techniques...",
            publication_year=2024,
            journal_id=journal.id
        )
        
        db.add(work)
        db.commit()
        print("✅ Work creation successful")
        
        # Test retrieval
        retrieved = db.query(Work).filter_by(openalex_id="test_work_001").first()
        if retrieved and retrieved.publication_year == 2024:
            print("✅ Work retrieval successful")
        else:
            print("❌ Work retrieval failed")
            return False
        
        # Test relationship
        if retrieved.journal and retrieved.journal.name == "Work Test Journal":
            print("✅ Work-Journal relationship successful")
        else:
            print("❌ Work-Journal relationship failed")
            return False
        
        # Cleanup
        db.delete(work)
        db.delete(journal)
        db.commit()
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Work operations failed: {e}")
        traceback.print_exc()
        return False

def test_query_and_recommendation_operations():
    """Test QueryRun and Recommendation models"""
    print("\n🔍 Testing Query and Recommendation operations...")
    try:
        db = SessionLocal()
        
        # Create journal and query
        journal = Journal(
            openalex_id="test_rec_journal_001",
            name="Recommendation Test Journal"
        )
        db.add(journal)
        db.commit()
        
        query_run = QueryRun(
            session_id="test-session-123",
            query_text="machine learning artificial intelligence",
            model_used="test_model",
            timestamp=datetime.utcnow()
        )
        db.add(query_run)
        db.commit()
        print("✅ QueryRun creation successful")
        
        # Create recommendation
        recommendation = Recommendation(
            query_id=query_run.id,
            journal_id=journal.id,
            similarity=0.85,
            rank=1
        )
        db.add(recommendation)
        db.commit()
        print("✅ Recommendation creation successful")
        
        # Test retrieval
        retrieved_query = db.query(QueryRun).filter_by(session_id="test-session-123").first()
        if retrieved_query and retrieved_query.query_text == "machine learning artificial intelligence":
            print("✅ QueryRun retrieval successful")
        else:
            print("❌ QueryRun retrieval failed")
            return False
        
        retrieved_rec = db.query(Recommendation).filter_by(query_id=query_run.id).first()
        if retrieved_rec and retrieved_rec.similarity == 0.85:
            print("✅ Recommendation retrieval successful")
        else:
            print("❌ Recommendation retrieval failed")
            return False
        
        # Cleanup
        db.delete(recommendation)
        db.delete(query_run)
        db.delete(journal)
        db.commit()
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Query/Recommendation operations failed: {e}")
        traceback.print_exc()
        return False

def test_database_performance():
    """Test database performance with bulk operations"""
    print("\n🔍 Testing database performance...")
    try:
        db = SessionLocal()
        
        # Bulk insert test
        journals = []
        for i in range(100):
            journal = Journal(
                openalex_id=f"perf_test_{i:03d}",
                name=f"Performance Test Journal {i}",
                display_name=f"Performance Test Journal {i}"
            )
            journals.append(journal)
        
        start_time = datetime.now()
        db.add_all(journals)
        db.commit()
        end_time = datetime.now()
        
        insert_time = (end_time - start_time).total_seconds()
        print(f"✅ Bulk insert of 100 journals: {insert_time:.3f} seconds")
        
        # Bulk query test
        start_time = datetime.now()
        results = db.query(Journal).filter(Journal.openalex_id.like("perf_test_%")).all()
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds()
        print(f"✅ Bulk query of 100 journals: {query_time:.3f} seconds")
        
        if len(results) == 100:
            print("✅ Bulk operations data integrity verified")
        else:
            print(f"❌ Data integrity issue: expected 100, got {len(results)}")
            return False
        
        # Cleanup
        for journal in results:
            db.delete(journal)
        db.commit()
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

def test_database_constraints():
    """Test database constraints and integrity"""
    print("\n🔍 Testing database constraints...")
    try:
        db = SessionLocal()
        
        # Test unique constraint on openalex_id
        journal1 = Journal(
            openalex_id="constraint_test_001",
            name="Constraint Test Journal 1"
        )
        journal2 = Journal(
            openalex_id="constraint_test_001",  # Same ID - should fail
            name="Constraint Test Journal 2"
        )
        
        db.add(journal1)
        db.commit()
        print("✅ First journal with unique ID created")
        
        try:
            db.add(journal2)
            db.commit()
            print("❌ Duplicate openalex_id constraint not enforced")
            return False
        except Exception:
            print("✅ Unique constraint on openalex_id enforced correctly")
            db.rollback()
        
        # Test foreign key constraint
        profile = JournalProfile(
            journal_id=99999,  # Non-existent journal ID
            scope_text="test scope"
        )
        
        try:
            db.add(profile)
            db.commit()
            print("❌ Foreign key constraint not enforced")
            return False
        except Exception:
            print("✅ Foreign key constraint enforced correctly")
            db.rollback()
        
        # Cleanup
        db.delete(journal1)
        db.commit()
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Constraint test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all database tests"""
    print("=" * 70)
    print("🗄️  DATABASE SYSTEM COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    tests = [
        test_database_connection,
        test_database_schema,
        test_journal_operations,
        test_journal_profile_operations,
        test_work_operations,
        test_query_and_recommendation_operations,
        test_database_performance,
        test_database_constraints,
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
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"🏁 DATABASE TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL DATABASE TESTS PASSED! Database system is fully functional.")
        print("📊 Database features verified:")
        print("   ✅ Connection and schema")
        print("   ✅ CRUD operations")
        print("   ✅ Relationships and foreign keys")
        print("   ✅ Data integrity and constraints")
        print("   ✅ Performance with bulk operations")
        print("   ✅ JSON data storage and retrieval")
    else:
        print("⚠️  Some database tests failed. Please check the issues above.")
    
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
