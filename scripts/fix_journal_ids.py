import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import time
from app.models.base import SessionLocal
from app.models.entities import Journal
from app.core.config import settings

def search_openalex_id(journal_name):
    """Search OpenAlex API for the correct journal ID"""
    url = "https://api.openalex.org/sources"
    params = {
        "search": journal_name,
        "per-page": 5,
        "mailto": settings.OPENALEX_EMAIL
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if results:
            # Get the best match (first result)
            best_match = results[0]
            openalex_id = best_match["id"].split("/")[-1]
            display_name = best_match["display_name"]
            print(f"  Found: {display_name} -> {openalex_id}")
            return openalex_id
        else:
            print(f"  No results found for: {journal_name}")
            return None
    except Exception as e:
        print(f"  Error searching: {e}")
        return None

def fix_journal_ids():
    """Fix all journals with HIGH_QUALITY_ or MANUAL_ prefix"""
    db = SessionLocal()
    
    # Get all journals with non-standard IDs
    journals = db.query(Journal).filter(
        (Journal.openalex_id.like('HIGH_QUALITY_%')) | 
        (Journal.openalex_id.like('MANUAL_%'))
    ).all()
    
    print(f"Found {len(journals)} journals to fix\n")
    
    fixed = 0
    failed = []
    
    for journal in journals:
        print(f"Processing: {journal.name}")
        
        # Search for proper OpenAlex ID
        new_id = search_openalex_id(journal.name)
        
        if new_id:
            # Check if this ID already exists
            existing = db.query(Journal).filter_by(openalex_id=new_id).first()
            if existing and existing.id != journal.id:
                print(f"  WARNING: ID {new_id} already exists for '{existing.name}'")
                print(f"  Using synthetic ID instead")
                new_id = f"synthetic_{journal.id}"
            
            old_id = journal.openalex_id
            journal.openalex_id = new_id
            journal.source_type = "openalex" if new_id.startswith("S") else "manual"
            
            print(f"  ✓ Updated: {old_id} -> {new_id}\n")
            fixed += 1
        else:
            # Use synthetic ID as fallback
            new_id = f"synthetic_{journal.id}"
            old_id = journal.openalex_id
            journal.openalex_id = new_id
            journal.source_type = "manual"
            
            print(f"  ⚠ Using synthetic ID: {old_id} -> {new_id}\n")
            failed.append(journal.name)
            fixed += 1
        
        time.sleep(0.3)  # Be polite to the API
    
    # Commit all changes
    db.commit()
    print(f"\n{'='*60}")
    print(f"✓ Fixed {fixed} journal IDs")
    
    if failed:
        print(f"\n⚠ The following {len(failed)} journals used synthetic IDs:")
        for name in failed:
            print(f"  - {name}")
    
    print(f"\nAll journal IDs are now uniform!")
    db.close()

if __name__ == "__main__":
    fix_journal_ids()
