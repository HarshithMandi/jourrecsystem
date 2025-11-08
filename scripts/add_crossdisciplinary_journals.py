import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests, json, time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.entities import Journal, JournalProfile
from app.core.config import settings

# Use the cross-disciplinary database
DATABASE_URL = "sqlite:///./data/journal_rec_crossdisciplinary.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define diverse research fields to fetch journals from
RESEARCH_FIELDS = {
    "Biology": [
        "botany", "zoology", "ecology", "marine biology", "microbiology",
        "genetics", "molecular biology", "cell biology", "evolutionary biology"
    ],
    "Environmental Sciences": [
        "environmental science", "climate change", "conservation biology",
        "sustainability", "renewable energy", "ecology", "biodiversity"
    ],
    "Medicine & Health": [
        "medicine", "public health", "epidemiology", "pharmacology",
        "neuroscience", "cardiology", "oncology", "immunology"
    ],
    "Physical Sciences": [
        "physics", "chemistry", "materials science", "astronomy",
        "geophysics", "atmospheric science", "quantum physics"
    ],
    "Engineering": [
        "mechanical engineering", "civil engineering", "electrical engineering",
        "chemical engineering", "biomedical engineering", "aerospace engineering"
    ],
    "Social Sciences": [
        "psychology", "sociology", "economics", "political science",
        "anthropology", "education", "linguistics"
    ],
    "Agriculture & Food": [
        "agriculture", "agronomy", "horticulture", "food science",
        "veterinary science", "animal science", "forestry"
    ],
    "Earth Sciences": [
        "geology", "oceanography", "meteorology", "hydrology",
        "paleontology", "seismology", "volcanology"
    ]
}

def search_journals_by_topic(topic, per_page=50):
    """Search for journals by specific topic using OpenAlex API"""
    url = "https://api.openalex.org/sources"
    
    # Search for journals that are active and have reasonable citation counts
    params = {
        "search": topic,
        "filter": "type:journal,works_count:>100",  # Only journals with substantial works
        "per-page": per_page,
        "mailto": settings.OPENALEX_EMAIL,
        "sort": "cited_by_count:desc"  # Prioritize well-cited journals
    }
    
    print(f"  Searching for '{topic}' journals...")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        print(f"  Error searching for {topic}: {e}")
        return []

def ingest_cross_disciplinary_journals():
    """Ingest journals from various research fields"""
    session = SessionLocal()
    
    # Get initial count
    initial_count = session.query(Journal).count()
    print(f"Initial journals in database: {initial_count}")
    
    total_added = 0
    total_updated = 0
    
    for field, topics in RESEARCH_FIELDS.items():
        print(f"\n{'='*60}")
        print(f"Processing {field} journals...")
        print(f"{'='*60}")
        
        field_added = 0
        field_updated = 0
        
        for topic in topics:
            journals = search_journals_by_topic(topic, per_page=30)
            
            for src in journals:
                try:
                    openalex_id = src["id"].split("/")[-1]
                    
                    # Check if journal already exists
                    existing = session.query(Journal).filter_by(openalex_id=openalex_id).first()
                    
                    if existing:
                        # Update existing journal
                        existing.name = src["display_name"]
                        existing.display_name = src["display_name"]
                        existing.issn = src.get("issn_l")
                        existing.eissn = src.get("issn", [None])[0] if src.get("issn") else None
                        existing.is_open_access = src.get("is_oa", False)
                        existing.publisher = src.get("host_organization")
                        existing.subjects = json.dumps(src.get("topics", []))
                        field_updated += 1
                    else:
                        # Create new journal
                        journal = Journal(
                            openalex_id=openalex_id,
                            source_type="openalex",
                            name=src["display_name"],
                            display_name=src["display_name"],
                            issn=src.get("issn_l"),
                            eissn=src.get("issn", [None])[0] if src.get("issn") else None,
                            is_open_access=src.get("is_oa", False),
                            publisher=src.get("host_organization"),
                            subjects=json.dumps(src.get("topics", []))
                        )
                        session.add(journal)
                        field_added += 1
                        
                        # Create empty profile for the new journal
                        profile = JournalProfile(journal=journal)
                        session.add(profile)
                        
                except Exception as e:
                    print(f"  Error processing journal {src.get('display_name', 'Unknown')}: {e}")
                    continue
            
            # Commit after each topic to save progress
            try:
                session.commit()
                print(f"  ✓ {topic}: +{field_added} new, ~{field_updated} updated")
            except Exception as e:
                print(f"  Error committing {topic}: {e}")
                session.rollback()
            
            # Rate limiting
            time.sleep(0.3)
        
        total_added += field_added
        total_updated += field_updated
        print(f"  {field} Summary: {field_added} new journals, {field_updated} updated")
    
    # Final count
    final_count = session.query(Journal).count()
    session.close()
    
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Initial journals: {initial_count}")
    print(f"Final journals: {final_count}")
    print(f"New journals added: {total_added}")
    print(f"Journals updated: {total_updated}")
    print(f"Net increase: {final_count - initial_count}")
    
    return final_count - initial_count

def show_sample_journals():
    """Display sample journals from different fields"""
    session = SessionLocal()
    
    print(f"\n{'='*60}")
    print("SAMPLE JOURNALS BY FIELD")
    print(f"{'='*60}")
    
    # Get diverse sample by searching for specific keywords in journal names
    sample_keywords = [
        "botany", "zoology", "environmental", "physics", "chemistry",
        "psychology", "engineering", "agriculture", "geology", "medicine"
    ]
    
    for keyword in sample_keywords:
        journals = session.query(Journal).filter(
            Journal.name.ilike(f"%{keyword}%")
        ).limit(3).all()
        
        if journals:
            print(f"\n{keyword.upper()} related journals:")
            for j in journals:
                oa_status = "🟢 OA" if j.is_open_access else "🔴 Closed"
                publisher = j.publisher or "Unknown"
                print(f"  - {j.name}")
                print(f"    {oa_status} | Publisher: {publisher}")
    
    session.close()

if __name__ == "__main__":
    print("="*60)
    print("CROSS-DISCIPLINARY JOURNAL INGESTION")
    print("="*60)
    print("Target database: data/journal_rec_crossdisciplinary.db")
    print()
    
    # Ingest journals
    new_count = ingest_cross_disciplinary_journals()
    
    # Show samples
    show_sample_journals()
    
    print(f"\n{'='*60}")
    print("To use this database, update your config or rename the file")
    print(f"{'='*60}")
