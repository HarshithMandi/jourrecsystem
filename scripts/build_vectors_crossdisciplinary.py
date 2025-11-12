import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.entities import Journal, JournalProfile

# Use the cross-disciplinary database
DATABASE_URL = "sqlite:///./data/journal_rec_crossdisciplinary.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define local model path
MODEL_DIR = project_root / "models" / "all-MiniLM-L6-v2"

print("Loading models...")
# Load BERT model from local directory if it exists, otherwise download
if MODEL_DIR.exists():
    bert_general = SentenceTransformer(str(MODEL_DIR))
    print(f"✓ Loaded BERT model from local directory: {MODEL_DIR}")
else:
    print("⚠ Downloading model (first time only)...")
    bert_general = SentenceTransformer("all-MiniLM-L6-v2")
    # Save model locally for future offline use
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    bert_general.save(str(MODEL_DIR))
    print(f"✓ Model saved to: {MODEL_DIR}")
print("✓ BERT model loaded (384 dimensions)")

def build_vectors():
    """Rebuild BERT and TF-IDF vectors for all journals in the cross-disciplinary database"""
    db = SessionLocal()
    
    # Get all journals
    journals = db.query(Journal).all()
    print(f"\nFound {len(journals)} journals in database")
    
    if not journals:
        print("No journals found!")
        db.close()
        return
    
    # Build corpus for TF-IDF (same format as original)
    print("\nBuilding TF-IDF corpus...")
    corpus = []
    for j in journals:
        text = j.name + " " + (j.publisher or "")
        corpus.append(text)
    
    # Fit TF-IDF - use same max_features to ensure consistency
    # Even though we have more journals, keep 20K features for compatibility
    tfidf = TfidfVectorizer(max_features=20_000, stop_words="english")
    tfidf.fit(corpus)
    actual_features = len(tfidf.get_feature_names_out())
    print(f"✓ TF-IDF fitted on {len(corpus)} journals (max {actual_features} features)")
    
    # Clear ALL existing vectors first to ensure consistency
    print("\nClearing all existing vectors...")
    for j in journals:
        if j.profile:
            j.profile.bert_vector = None
            j.profile.tfidf_vector = None
    db.commit()
    print("✓ All vectors cleared")
    
    # Build vectors for each journal
    print("\nBuilding vectors for each journal...")
    success_count = 0
    error_count = 0
    
    for idx, j in enumerate(journals, 1):
        try:
            # Get or create profile
            profile = j.profile
            if not profile:
                profile = JournalProfile(journal_id=j.id)
                db.add(profile)
                db.flush()  # Get the profile ID
            
            # Build BERT vector (384-dim)
            text_for_embedding = j.name + " " + (j.publisher or "")
            vec_bert = bert_general.encode([text_for_embedding])[0]
            profile.bert_vector = json.dumps(vec_bert.tolist())
            
            # Build TF-IDF vector (20,000-dim)
            text_for_tfidf = j.name + " " + (j.publisher or "")
            vec_tfidf_sparse = tfidf.transform([text_for_tfidf])
            vec_tfidf = np.array(vec_tfidf_sparse.todense()).flatten()
            profile.tfidf_vector = json.dumps(vec_tfidf.tolist())
            
            success_count += 1
            
            # Progress indicator
            if idx % 50 == 0:
                print(f"  Processed {idx}/{len(journals)} journals...")
                db.commit()  # Commit in batches
        
        except Exception as e:
            print(f"  Error processing {j.name}: {e}")
            error_count += 1
            continue
    
    # Final commit
    db.commit()
    print(f"\n✓ Vector building complete!")
    print(f"  Success: {success_count} journals")
    print(f"  Errors: {error_count} journals")
    
    # Verify vectors
    print("\nVerifying vectors...")
    journals_with_vectors = db.query(Journal).join(JournalProfile).filter(
        JournalProfile.bert_vector.isnot(None),
        JournalProfile.tfidf_vector.isnot(None)
    ).count()
    
    print(f"✓ {journals_with_vectors}/{len(journals)} journals have complete vectors")
    
    # Show sample vectors
    print("\nSample vector dimensions:")
    sample = db.query(Journal).join(JournalProfile).filter(
        JournalProfile.bert_vector.isnot(None)
    ).first()
    
    if sample and sample.profile:
        bert_vec = json.loads(sample.profile.bert_vector)
        tfidf_vec = json.loads(sample.profile.tfidf_vector)
        print(f"  Journal: {sample.name}")
        print(f"  BERT vector: {len(bert_vec)} dimensions")
        print(f"  TF-IDF vector: {len(tfidf_vec)} dimensions")
    
    db.close()
    
    print(f"\n{'='*60}")
    print("VECTOR BUILDING COMPLETE")
    print(f"{'='*60}")
    print(f"Database: data/journal_rec_crossdisciplinary.db")
    print(f"Total journals: {len(journals)}")
    print(f"Journals with vectors: {journals_with_vectors}")
    print(f"\nTo use this database:")
    print(f"1. Test it first to make sure it works")
    print(f"2. If satisfied, rename:")
    print(f"   journal_rec.db -> journal_rec_backup_original.db")
    print(f"   journal_rec_crossdisciplinary.db -> journal_rec.db")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("="*60)
    print("REBUILDING VECTORS FOR CROSS-DISCIPLINARY DATABASE")
    print("="*60)
    build_vectors()
