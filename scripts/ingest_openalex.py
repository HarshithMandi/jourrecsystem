import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests, json, time
from app.models.base import SessionLocal
from app.models.entities import Journal, Work
from app.core.config import settings

def ingest_journals(cursor=None, per_page=200):
    """Ingest journals from OpenAlex API"""
    url = "https://api.openalex.org/sources"
    params = {"per-page": per_page, "mailto": settings.OPENALEX_EMAIL}
    
    if cursor and cursor != "*":
        params["cursor"] = cursor
    
    print(f"Fetching journals with cursor: {cursor}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    session = SessionLocal()
    count = 0
    
    for src in data.get("results", []):
        try:
            journal = Journal(
                openalex_id=src["id"].split("/")[-1],
                name=src["display_name"],
                display_name=src["display_name"],
                issn=src.get("issn_l"),
                is_open_access=src.get("is_oa", False),
                publisher=src.get("publisher"),
                subjects=json.dumps(src.get("topics", []))  # Store as JSON string
            )
            session.merge(journal)
            count += 1
        except Exception as e:
            print(f"Error processing journal {src.get('display_name', 'Unknown')}: {e}")
            continue
    
    session.commit()
    session.close()
    
    print(f"Processed {count} journals")
    return data["meta"].get("next_cursor")

def ingest_works_for_journal(journal_id, max_works=50):
    """Ingest works for a specific journal"""
    url = f"https://api.openalex.org/works"
    params = {
        "filter": f"primary_location.source.id:https://openalex.org/S{journal_id}",
        "per-page": max_works, 
        "mailto": settings.OPENALEX_EMAIL
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        session = SessionLocal()
        journal = session.query(Journal).filter_by(openalex_id=journal_id).first()
        
        if not journal:
            print(f"Journal {journal_id} not found in database")
            session.close()
            return
        
        count = 0
        for work in data.get("results", []):
            try:
                w = Work(
                    openalex_id=work["id"].split("/")[-1],
                    title=work.get("title", ""),
                    abstract=work.get("abstract"),
                    publication_year=work.get("publication_year"),
                    journal_id=journal.id
                )
                session.merge(w)
                count += 1
            except Exception as e:
                print(f"Error processing work {work.get('title', 'Unknown')}: {e}")
                continue
        
        session.commit()
        session.close()
        print(f"Processed {count} works for journal {journal.name}")
        
    except Exception as e:
        print(f"Error fetching works for journal {journal_id}: {e}")

if __name__ == "__main__":
    print("Starting OpenAlex data ingestion...")
    
    # Check current journal count
    session = SessionLocal()
    initial_count = session.query(Journal).count()
    session.close()
    print(f"Initial journals in database: {initial_count}")
    
    # Ingest journals first
    cursor = None
    page = 1
    max_pages = 3  # Limit to prevent too much data for testing
    
    print(f"Ingesting journals (max {max_pages} pages)...")
    while page <= max_pages:
        print(f"Processing page {page}/{max_pages}")
        try:
            new_cursor = ingest_journals(cursor)
            if new_cursor is None:
                print("No more pages available")
                break
            cursor = new_cursor
            page += 1
            time.sleep(1)  # Be polite to the API
        except Exception as e:
            print(f"Error during ingestion: {e}")
            break
    
    # Check final journal count
    session = SessionLocal()
    final_count = session.query(Journal).count()
    print(f"Final journals in database: {final_count}")
    print(f"New journals added: {final_count - initial_count}")
    
    # Show some example journals
    sample_journals = session.query(Journal).limit(5).all()
    print("\nSample journals in database:")
    for journal in sample_journals:
        print(f"  - {journal.name} (ID: {journal.openalex_id})")
    
    session.close()
    print("Ingestion completed!")
