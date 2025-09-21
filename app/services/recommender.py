import json, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings
from app.models.base import SessionLocal
from app.models.entities import Journal, JournalProfile, QueryRun, Recommendation
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# load models once
bert = SentenceTransformer("all-MiniLM-L6-v2")
tfidf = TfidfVectorizer(max_features=20_000, stop_words="english")

# Initialize TF-IDF on a simple corpus to avoid empty data issues
_session = SessionLocal()
try:
    corpus = []
    for j, p in _session.query(Journal, JournalProfile).outerjoin(JournalProfile).all():
        scope_text = getattr(p, 'scope_text', None) if p else None
        text = scope_text or j.name or j.display_name or "unknown"
        corpus.append(text)
    
    if corpus:
        tfidf.fit(corpus)
    else:
        # Fallback corpus for empty database
        tfidf.fit(["machine learning", "data science", "computer science"])
except Exception as e:
    print(f"Warning: Could not initialize TF-IDF with database corpus: {e}")
    # Fallback corpus
    tfidf.fit(["machine learning", "data science", "computer science"])
finally:
    _session.close()

def rank_journals(abstract: str, top_k: int = settings.TOP_K):
    db = SessionLocal()

    # encode query
    vec_tfidf_sparse = tfidf.transform([abstract])
    vec_tfidf = np.array(vec_tfidf_sparse.todense()).flatten()  # Convert sparse to dense properly
    vec_bert = bert.encode([abstract])[0]

    # fetch candidates
    journals = db.query(Journal).join(JournalProfile).all()
    sims = []
    for j in journals:
        p = j.profile
        if not p or not p.tfidf_vector or not p.bert_vector:
            continue  # Skip journals without vectors
            
        try:
            v_tfidf = np.array(json.loads(p.tfidf_vector))
            v_bert = np.array(json.loads(p.bert_vector))
            
            # Calculate similarities using numpy dot product (normalized)
            def cosine_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            sim_tfidf = cosine_sim(vec_tfidf, v_tfidf)
            sim_bert = cosine_sim(vec_bert, v_bert)
            sim = 0.5 * sim_tfidf + 0.5 * sim_bert
            
            sims.append((j, sim))
        except (json.JSONDecodeError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Could not parse vectors for journal {j.name}: {e}")
            continue

    sims.sort(key=lambda x: x[1], reverse=True)
    ranked = sims[:top_k]

    # audit trail
    q = QueryRun(query_text=abstract, model_used="ensemble")
    db.add(q); db.commit()

    for rank, (j, score) in enumerate(ranked, 1):
        db.add(Recommendation(query_id=q.id, journal_id=j.id,
                              similarity=score, rank=rank))
    db.commit()

    results = [{"journal": j.name, "similarity": round(score,3)} 
               for j,score in ranked]
    db.close()
    return results
