import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tqdm, json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from app.models.base import SessionLocal
from app.models.entities import Journal, JournalProfile
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

# Define local model path
MODEL_DIR = project_root / "models" / "all-MiniLM-L6-v2"

tfidf = TfidfVectorizer(max_features=20_000, stop_words="english")

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

def main():
    db = SessionLocal()

    # 1. Build corpus
    scopes = []
    ids = []
    for j in db.scalars(select(Journal)).all():
        scopes.append(j.name + " " + (j.publisher or ""))
        ids.append(j.id)

    # 2. TF-IDF
    tfidf_mat = tfidf.fit_transform(scopes)

    # 3. Store vectors
    tfidf_dense = tfidf_mat.todense()  # Convert entire matrix to dense first
    for idx, jid in enumerate(ids):
        vec_dense = np.array(tfidf_dense[idx]).flatten()  # Get row from dense matrix
        profile = db.query(JournalProfile).filter_by(journal_id=jid).first()
        if not profile:
            profile = JournalProfile(journal_id=jid)
            db.add(profile)
        profile.tfidf_vector = json.dumps(vec_dense.tolist())
    db.commit()

    # 4. BERT embeddings - general model (batch)
    print("Encoding with general BERT model (all-MiniLM-L6-v2, 384 dimensions)...")
    batch = 512
    for start in tqdm.trange(0, len(scopes), batch):
        embeds = bert_general.encode(scopes[start:start+batch]).tolist()
        for offset, vec in enumerate(embeds):
            jid = ids[start+offset]
            profile = db.query(JournalProfile).filter_by(journal_id=jid).first()
            if profile:  # Make sure profile exists
                profile.bert_vector = json.dumps(vec)
    db.commit()
    db.close()
    
    print(f"\n✓ Successfully rebuilt vectors for {len(ids)} journals")
    print("  - TF-IDF vectors: 20,000 features")
    print("  - General BERT vectors: 384 dimensions (all-MiniLM-L6-v2)")

if __name__ == "__main__":
    main()
