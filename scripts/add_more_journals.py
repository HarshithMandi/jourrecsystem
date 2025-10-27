#!/usr/bin/env python3
"""
Add More Journals to Database
Script to safely add additional journals from OpenAlex API
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests, json, time
from app.models.base import SessionLocal
from app.models.entities import Journal
from app.core.config import settings

def get_existing_journal_ids():
    """Get all existing journal OpenAlex IDs from database"""
    session = SessionLocal()
    existing_ids = set()
    try:
        journals = session.query(Journal.openalex_id).all()
        existing_ids = {j.openalex_id for j in journals}
        print(f"Found {len(existing_ids)} existing journals in database")
    finally:
        session.close()
    return existing_ids

def add_journals_from_openalex(target_count=500, per_page=200):
    """Add journals from OpenAlex up to target count"""
    existing_ids = get_existing_journal_ids()
    
    session = SessionLocal()
    current_count = session.query(Journal).count()
    session.close()
    
    print(f"Current journal count: {current_count}")
    print(f"Target journal count: {target_count}")
    
    if current_count >= target_count:
        print("Target already reached!")
        return
    
    needed = target_count - current_count
    print(f"Need to add approximately {needed} more journals")
    
    url = "https://api.openalex.org/sources"
    cursor = None
    added_count = 0
    page = 1
    
    while added_count < needed:
        params = {
            "per-page": per_page,
            "mailto": settings.OPENALEX_EMAIL or "example@email.com"
        }
        
        if cursor and cursor != "*":
            params["cursor"] = cursor
        
        print(f"Fetching page {page}, cursor: {cursor}")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
        
        session = SessionLocal()
        page_added = 0
        
        for src in data.get("results", []):
            openalex_id = src["id"].split("/")[-1]
            
            # Skip if already exists
            if openalex_id in existing_ids:
                continue
            
            try:
                # Filter for journals with meaningful data
                if (src.get("display_name") and 
                    src.get("display_name") not in ["Unknown", "N/A", ""] and
                    src.get("type") == "journal"):
                    
                    journal = Journal(
                        openalex_id=openalex_id,
                        name=src["display_name"],
                        display_name=src["display_name"],
                        issn=src.get("issn_l"),
                        is_open_access=src.get("is_oa", False),
                        publisher=src.get("host_organization", {}).get("display_name") if src.get("host_organization") else None,
                        subjects=json.dumps(src.get("topics", []))
                    )
                    
                    session.add(journal)
                    existing_ids.add(openalex_id)  # Add to our tracking set
                    page_added += 1
                    added_count += 1
                    
                    if added_count >= needed:
                        break
                        
            except Exception as e:
                print(f"Error processing journal {src.get('display_name', 'Unknown')}: {e}")
                continue
        
        try:
            session.commit()
            print(f"Page {page}: Added {page_added} new journals (Total added: {added_count})")
        except Exception as e:
            print(f"Error committing page {page}: {e}")
            session.rollback()
        finally:
            session.close()
        
        # Get next cursor
        cursor = data["meta"].get("next_cursor")
        if not cursor:
            print("No more pages available")
            break
            
        page += 1
        time.sleep(0.5)  # Be polite to the API
    
    print(f"Completed! Added {added_count} new journals")

def add_specific_high_quality_journals():
    """Add some high-quality, well-known journals manually"""
    high_quality_journals = [
        {
            "name": "Nature Machine Intelligence",
            "display_name": "Nature Machine Intelligence",
            "publisher": "Nature Publishing Group",
            "is_open_access": False,
            "impact_factor": 25.8,
            "subjects": json.dumps([{"field": "Computer Science", "subfield": "Machine Learning"}])
        },
        {
            "name": "Journal of Machine Learning Research",
            "display_name": "Journal of Machine Learning Research",
            "publisher": "MIT Press",
            "is_open_access": True,
            "impact_factor": 4.3,
            "subjects": json.dumps([{"field": "Computer Science", "subfield": "Machine Learning"}])
        },
        {
            "name": "Nature Biotechnology",
            "display_name": "Nature Biotechnology",
            "publisher": "Nature Publishing Group",
            "is_open_access": False,
            "impact_factor": 68.2,
            "subjects": json.dumps([{"field": "Biology", "subfield": "Biotechnology"}])
        },
        {
            "name": "Cell",
            "display_name": "Cell",
            "publisher": "Elsevier",
            "is_open_access": False,
            "impact_factor": 66.9,
            "subjects": json.dumps([{"field": "Biology", "subfield": "Cell Biology"}])
        },
        {
            "name": "Proceedings of the National Academy of Sciences",
            "display_name": "Proceedings of the National Academy of Sciences",
            "publisher": "National Academy of Sciences",
            "is_open_access": False,
            "impact_factor": 12.8,
            "subjects": json.dumps([{"field": "Multidisciplinary", "subfield": "General"}])
        }
    ]
    
    session = SessionLocal()
    added = 0
    
    for journal_data in high_quality_journals:
        try:
            # Check if journal already exists by name
            existing = session.query(Journal).filter_by(name=journal_data["name"]).first()
            if existing:
                print(f"Journal '{journal_data['name']}' already exists")
                continue
            
            journal = Journal(
                openalex_id=f"MANUAL_{journal_data['name'].replace(' ', '_').upper()}",
                **journal_data
            )
            session.add(journal)
            added += 1
            print(f"Added high-quality journal: {journal_data['name']}")
            
        except Exception as e:
            print(f"Error adding journal {journal_data['name']}: {e}")
    
    try:
        session.commit()
        print(f"Successfully added {added} high-quality journals")
    except Exception as e:
        print(f"Error committing high-quality journals: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    print("=== Adding More Journals to Database ===")
    
    # Check current state
    session = SessionLocal()
    initial_count = session.query(Journal).count()
    session.close()
    print(f"Starting with {initial_count} journals")
    
    # Add high-quality journals first
    print("\n1. Adding high-quality journals...")
    add_specific_high_quality_journals()
    
    # Add more from OpenAlex
    print("\n2. Adding journals from OpenAlex...")
    add_journals_from_openalex(target_count=800)  # Increase to 800 journals
    
    # Final count
    session = SessionLocal()
    final_count = session.query(Journal).count()
    session.close()
    
    print(f"\n=== Summary ===")
    print(f"Initial journals: {initial_count}")
    print(f"Final journals: {final_count}")
    print(f"New journals added: {final_count - initial_count}")
    
    # Show some new journals
    if final_count > initial_count:
        session = SessionLocal()
        new_journals = session.query(Journal).order_by(Journal.id.desc()).limit(10).all()
        print(f"\nRecently added journals:")
        for journal in new_journals:
            print(f"  - {journal.name} ({journal.publisher or 'Unknown Publisher'})")
        session.close()
    
    print("\nDone! You may need to rebuild vectors for the new journals.")