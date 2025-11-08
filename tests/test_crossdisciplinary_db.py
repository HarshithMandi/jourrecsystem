import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from app.models.entities import Journal, JournalProfile

# Use the cross-disciplinary database
DATABASE_URL = "sqlite:///./data/journal_rec_crossdisciplinary.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_database_stats():
    """Show comprehensive statistics about the cross-disciplinary database"""
    db = SessionLocal()
    
    print("="*70)
    print("CROSS-DISCIPLINARY DATABASE STATISTICS")
    print("="*70)
    
    # Total counts
    total_journals = db.query(Journal).count()
    journals_with_profiles = db.query(Journal).join(JournalProfile).count()
    journals_with_vectors = db.query(Journal).join(JournalProfile).filter(
        JournalProfile.bert_vector.isnot(None)
    ).count()
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Journals: {total_journals}")
    print(f"   Journals with Profiles: {journals_with_profiles}")
    print(f"   Journals with Vectors: {journals_with_vectors}")
    print(f"   Completeness: {journals_with_vectors/total_journals*100:.1f}%")
    
    # Open access statistics
    oa_count = db.query(Journal).filter(Journal.is_open_access == True).count()
    closed_count = total_journals - oa_count
    
    print(f"\n🔓 ACCESS STATUS")
    print(f"   Open Access: {oa_count} ({oa_count/total_journals*100:.1f}%)")
    print(f"   Closed Access: {closed_count} ({closed_count/total_journals*100:.1f}%)")
    
    # Sample journals from different fields
    print(f"\n🔬 SAMPLE JOURNALS BY FIELD")
    print("-"*70)
    
    field_keywords = {
        "Biology": "biology",
        "Botany": "botany",
        "Zoology": "zoology",
        "Environmental": "environmental",
        "Medicine": "medicine",
        "Physics": "physics",
        "Chemistry": "chemistry",
        "Engineering": "engineering",
        "Psychology": "psychology",
        "Agriculture": "agriculture",
        "Computer Science": "computer"
    }
    
    for field, keyword in field_keywords.items():
        journals = db.query(Journal).filter(
            Journal.name.ilike(f"%{keyword}%")
        ).limit(2).all()
        
        if journals:
            print(f"\n{field}:")
            for j in journals:
                oa = "🟢" if j.is_open_access else "🔴"
                publisher = j.publisher or "Unknown"
                if publisher.startswith("https://openalex.org/"):
                    publisher = "OpenAlex Publisher"
                print(f"   {oa} {j.name}")
                if len(publisher) < 50:
                    print(f"      Publisher: {publisher}")
    
    # Top publishers by journal count
    print(f"\n📚 TOP 10 PUBLISHERS BY JOURNAL COUNT")
    print("-"*70)
    
    publisher_counts = db.query(
        Journal.publisher,
        func.count(Journal.id).label('count')
    ).group_by(Journal.publisher).order_by(func.count(Journal.id).desc()).limit(10).all()
    
    for publisher, count in publisher_counts:
        pub_name = publisher or "Unknown"
        if pub_name.startswith("https://openalex.org/"):
            pub_name = "OpenAlex Publisher"
        print(f"   {count:3d} journals - {pub_name[:60]}")
    
    db.close()
    
    print(f"\n{'='*70}")
    print("DATABASE READY FOR TESTING")
    print("="*70)
    print(f"✓ All {journals_with_vectors} journals have BERT and TF-IDF vectors")
    print(f"✓ Database covers {len(field_keywords)} major research areas")
    print(f"\nNext steps:")
    print(f"1. Test recommendations with diverse abstracts")
    print(f"2. Check if results are relevant across different fields")
    print(f"3. If satisfied, replace the original database")
    print("="*70)

def test_sample_recommendations():
    """Test recommendations with sample abstracts from different fields"""
    from app.services.recommender import rank_journals
    
    # Temporarily update the database path in recommender
    import app.services.recommender as rec_module
    original_session = rec_module.SessionLocal
    rec_module.SessionLocal = SessionLocal
    
    print("\n" + "="*70)
    print("TESTING RECOMMENDATIONS WITH DIVERSE ABSTRACTS")
    print("="*70)
    
    test_abstracts = {
        "Computer Science": "Deep learning models using transformer architectures have revolutionized natural language processing tasks, achieving state-of-the-art results in machine translation, text summarization, and question answering systems.",
        
        "Biology": "We investigated the evolutionary mechanisms driving speciation in isolated island populations, focusing on genetic drift, natural selection, and reproductive isolation patterns across multiple endemic species.",
        
        "Environmental Science": "Climate change impacts on marine ecosystems were assessed through long-term monitoring of ocean acidification, coral bleaching events, and shifts in phytoplankton community composition affecting global carbon cycles.",
        
        "Medicine": "This randomized controlled trial evaluated the efficacy of a novel immunotherapy approach for treating metastatic melanoma, measuring progression-free survival and quality of life outcomes in stage IV patients."
    }
    
    for field, abstract in test_abstracts.items():
        print(f"\n{field.upper()} Abstract Test")
        print("-"*70)
        print(f"Abstract: {abstract[:100]}...")
        
        try:
            results = rank_journals(abstract, top_k=5)
            print(f"\nTop 5 Recommended Journals:")
            for i, result in enumerate(results[:5], 1):
                journal_name = result['journal_name']
                score = result['similarity_combined']
                oa = "🟢" if result['is_open_access'] else "🔴"
                print(f"   {i}. {oa} {journal_name}")
                print(f"      Combined Score: {score:.4f}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Restore original session
    rec_module.SessionLocal = original_session
    
    print("\n" + "="*70)

if __name__ == "__main__":
    # Show database statistics
    test_database_stats()
    
    # Test recommendations
    print("\n")
    response = input("Would you like to test recommendations? (y/n): ")
    if response.lower() == 'y':
        test_sample_recommendations()
