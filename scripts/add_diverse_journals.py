#!/usr/bin/env python3
"""
Add Diverse High-Quality Journals
Script to add journals across different academic fields
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from app.models.base import SessionLocal
from app.models.entities import Journal

def add_comprehensive_journal_collection():
    """Add a comprehensive collection of high-quality journals across disciplines"""
    
    journals_to_add = [
        # Computer Science & AI
        {
            "name": "Artificial Intelligence",
            "display_name": "Artificial Intelligence",
            "publisher": "Elsevier",
            "is_open_access": False,
            "impact_factor": 8.1,
            "subjects": json.dumps([{"field": "Computer Science", "subfield": "Artificial Intelligence"}])
        },
        {
            "name": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
            "display_name": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
            "publisher": "IEEE",
            "is_open_access": False,
            "impact_factor": 17.7,
            "subjects": json.dumps([{"field": "Computer Science", "subfield": "Computer Vision"}])
        },
        {
            "name": "ACM Computing Surveys",
            "display_name": "ACM Computing Surveys",
            "publisher": "ACM",
            "is_open_access": False,
            "impact_factor": 16.6,
            "subjects": json.dumps([{"field": "Computer Science", "subfield": "General"}])
        },
        {
            "name": "Journal of the ACM",
            "display_name": "Journal of the ACM",
            "publisher": "ACM",
            "is_open_access": False,
            "impact_factor": 2.5,
            "subjects": json.dumps([{"field": "Computer Science", "subfield": "Theory"}])
        },
        
        # Medicine & Health
        {
            "name": "The Lancet",
            "display_name": "The Lancet",
            "publisher": "Elsevier",
            "is_open_access": False,
            "impact_factor": 202.7,
            "subjects": json.dumps([{"field": "Medicine", "subfield": "General Medicine"}])
        },
        {
            "name": "New England Journal of Medicine",
            "display_name": "New England Journal of Medicine",
            "publisher": "Massachusetts Medical Society",
            "is_open_access": False,
            "impact_factor": 176.1,
            "subjects": json.dumps([{"field": "Medicine", "subfield": "General Medicine"}])
        },
        {
            "name": "JAMA",
            "display_name": "JAMA",
            "publisher": "American Medical Association",
            "is_open_access": False,
            "impact_factor": 157.3,
            "subjects": json.dumps([{"field": "Medicine", "subfield": "General Medicine"}])
        },
        {
            "name": "Nature Medicine",
            "display_name": "Nature Medicine",
            "publisher": "Nature Publishing Group",
            "is_open_access": False,
            "impact_factor": 87.2,
            "subjects": json.dumps([{"field": "Medicine", "subfield": "Medical Research"}])
        },
        
        # Biology & Life Sciences
        {
            "name": "Science",
            "display_name": "Science",
            "publisher": "American Association for the Advancement of Science",
            "is_open_access": False,
            "impact_factor": 63.8,
            "subjects": json.dumps([{"field": "Multidisciplinary", "subfield": "General"}])
        },
        {
            "name": "Nature",
            "display_name": "Nature",
            "publisher": "Nature Publishing Group",
            "is_open_access": False,
            "impact_factor": 69.5,
            "subjects": json.dumps([{"field": "Multidisciplinary", "subfield": "General"}])
        },
        {
            "name": "Nature Genetics",
            "display_name": "Nature Genetics",
            "publisher": "Nature Publishing Group",
            "is_open_access": False,
            "impact_factor": 38.3,
            "subjects": json.dumps([{"field": "Biology", "subfield": "Genetics"}])
        },
        {
            "name": "Molecular Biology and Evolution",
            "display_name": "Molecular Biology and Evolution",
            "publisher": "Oxford University Press",
            "is_open_access": False,
            "impact_factor": 16.2,
            "subjects": json.dumps([{"field": "Biology", "subfield": "Evolutionary Biology"}])
        },
        
        # Physics & Engineering
        {
            "name": "Physical Review Letters",
            "display_name": "Physical Review Letters",
            "publisher": "American Physical Society",
            "is_open_access": False,
            "impact_factor": 9.2,
            "subjects": json.dumps([{"field": "Physics", "subfield": "General Physics"}])
        },
        {
            "name": "Nature Physics",
            "display_name": "Nature Physics",
            "publisher": "Nature Publishing Group",
            "is_open_access": False,
            "impact_factor": 19.6,
            "subjects": json.dumps([{"field": "Physics", "subfield": "General Physics"}])
        },
        {
            "name": "IEEE Transactions on Information Theory",
            "display_name": "IEEE Transactions on Information Theory",
            "publisher": "IEEE",
            "is_open_access": False,
            "impact_factor": 3.5,
            "subjects": json.dumps([{"field": "Engineering", "subfield": "Information Theory"}])
        },
        
        # Chemistry
        {
            "name": "Journal of the American Chemical Society",
            "display_name": "Journal of the American Chemical Society",
            "publisher": "American Chemical Society",
            "is_open_access": False,
            "impact_factor": 16.3,
            "subjects": json.dumps([{"field": "Chemistry", "subfield": "General Chemistry"}])
        },
        {
            "name": "Angewandte Chemie International Edition",
            "display_name": "Angewandte Chemie International Edition",
            "publisher": "Wiley",
            "is_open_access": False,
            "impact_factor": 16.8,
            "subjects": json.dumps([{"field": "Chemistry", "subfield": "General Chemistry"}])
        },
        
        # Social Sciences & Economics
        {
            "name": "American Economic Review",
            "display_name": "American Economic Review",
            "publisher": "American Economic Association",
            "is_open_access": False,
            "impact_factor": 9.5,
            "subjects": json.dumps([{"field": "Economics", "subfield": "General Economics"}])
        },
        {
            "name": "Quarterly Journal of Economics",
            "display_name": "Quarterly Journal of Economics",
            "publisher": "Oxford University Press",
            "is_open_access": False,
            "impact_factor": 11.1,
            "subjects": json.dumps([{"field": "Economics", "subfield": "Economic Theory"}])
        },
        {
            "name": "American Political Science Review",
            "display_name": "American Political Science Review",
            "publisher": "Cambridge University Press",
            "is_open_access": False,
            "impact_factor": 5.7,
            "subjects": json.dumps([{"field": "Political Science", "subfield": "General"}])
        },
        
        # Environmental Sciences
        {
            "name": "Environmental Science & Technology",
            "display_name": "Environmental Science & Technology",
            "publisher": "American Chemical Society",
            "is_open_access": False,
            "impact_factor": 11.4,
            "subjects": json.dumps([{"field": "Environmental Science", "subfield": "General"}])
        },
        {
            "name": "Nature Climate Change",
            "display_name": "Nature Climate Change",
            "publisher": "Nature Publishing Group",
            "is_open_access": False,
            "impact_factor": 30.7,
            "subjects": json.dumps([{"field": "Environmental Science", "subfield": "Climate Science"}])
        },
        
        # Mathematics
        {
            "name": "Annals of Mathematics",
            "display_name": "Annals of Mathematics",
            "publisher": "Princeton University",
            "is_open_access": False,
            "impact_factor": 4.8,
            "subjects": json.dumps([{"field": "Mathematics", "subfield": "Pure Mathematics"}])
        },
        {
            "name": "Journal of the American Mathematical Society",
            "display_name": "Journal of the American Mathematical Society",
            "publisher": "American Mathematical Society",
            "is_open_access": False,
            "impact_factor": 4.9,
            "subjects": json.dumps([{"field": "Mathematics", "subfield": "Pure Mathematics"}])
        },
        
        # Open Access Journals
        {
            "name": "PLOS ONE",
            "display_name": "PLOS ONE",
            "publisher": "Public Library of Science",
            "is_open_access": True,
            "impact_factor": 3.7,
            "subjects": json.dumps([{"field": "Multidisciplinary", "subfield": "General"}])
        },
        {
            "name": "Scientific Reports",
            "display_name": "Scientific Reports",
            "publisher": "Nature Publishing Group",
            "is_open_access": True,
            "impact_factor": 4.4,
            "subjects": json.dumps([{"field": "Multidisciplinary", "subfield": "General"}])
        },
        {
            "name": "eLife",
            "display_name": "eLife",
            "publisher": "eLife Sciences Publications",
            "is_open_access": True,
            "impact_factor": 8.7,
            "subjects": json.dumps([{"field": "Biology", "subfield": "General Biology"}])
        },
        
        # Technology & Innovation
        {
            "name": "Nature Nanotechnology",
            "display_name": "Nature Nanotechnology",
            "publisher": "Nature Publishing Group",
            "is_open_access": False,
            "impact_factor": 38.3,
            "subjects": json.dumps([{"field": "Nanotechnology", "subfield": "General"}])
        },
        {
            "name": "Advanced Materials",
            "display_name": "Advanced Materials",
            "publisher": "Wiley",
            "is_open_access": False,
            "impact_factor": 32.1,
            "subjects": json.dumps([{"field": "Materials Science", "subfield": "General"}])
        }
    ]
    
    session = SessionLocal()
    added_count = 0
    
    for journal_data in journals_to_add:
        try:
            # Check if journal already exists by name
            existing = session.query(Journal).filter_by(name=journal_data["name"]).first()
            if existing:
                print(f"Journal '{journal_data['name']}' already exists")
                continue
            
            journal = Journal(
                openalex_id=f"HIGH_QUALITY_{journal_data['name'].replace(' ', '_').replace('&', 'and').upper()}",
                **journal_data
            )
            session.add(journal)
            added_count += 1
            print(f"✓ Added: {journal_data['name']} (Impact Factor: {journal_data.get('impact_factor', 'N/A')})")
            
        except Exception as e:
            print(f"✗ Error adding journal {journal_data['name']}: {e}")
    
    try:
        session.commit()
        print(f"\n🎉 Successfully added {added_count} high-quality journals to the database!")
    except Exception as e:
        print(f"Error committing journals: {e}")
        session.rollback()
    finally:
        session.close()
    
    return added_count

if __name__ == "__main__":
    print("=== Adding High-Quality Journals Across All Disciplines ===\n")
    
    # Check current state
    session = SessionLocal()
    initial_count = session.query(Journal).count()
    session.close()
    print(f"Starting with {initial_count} journals in database\n")
    
    # Add the comprehensive collection
    added = add_comprehensive_journal_collection()
    
    # Final count
    session = SessionLocal()
    final_count = session.query(Journal).count()
    session.close()
    
    print(f"\n=== Summary ===")
    print(f"Initial journals: {initial_count}")
    print(f"Final journals: {final_count}")
    print(f"New journals added: {final_count - initial_count}")
    
    # Show sample by field
    if final_count > initial_count:
        session = SessionLocal()
        print(f"\n📊 Journal Distribution by Field:")
        
        # Get some examples from different fields
        fields = ["Computer Science", "Medicine", "Biology", "Physics", "Chemistry", "Economics"]
        for field in fields:
            journals = session.query(Journal).filter(Journal.subjects.like(f'%{field}%')).limit(3).all()
            if journals:
                print(f"\n{field}:")
                for journal in journals:
                    print(f"  - {journal.name}")
        
        session.close()
    
    print(f"\n✨ Database now contains {final_count} journals across multiple disciplines!")
    print("🔄 You may want to rebuild the ML vectors to include these new journals.")