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

# Initialize TF-IDF using the SAME corpus format as build_vectors.py
_session = SessionLocal()
try:
    corpus = []
    for j in _session.query(Journal).all():
        # Use same format as build_vectors.py: name + publisher
        text = j.name + " " + (j.publisher or "")
        corpus.append(text)
    
    if corpus:
        tfidf.fit(corpus)
        print(f"TF-IDF fitted on {len(corpus)} journals")
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
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 0.0  # Return 0 similarity for zero vectors
                return np.dot(a, b) / (norm_a * norm_b)
            
            sim_tfidf = cosine_sim(vec_tfidf, v_tfidf)
            sim_bert = cosine_sim(vec_bert, v_bert)
            # Weighted combination: 30% TF-IDF + 70% BERT for better semantic matching
            sim_combined = 0.3 * sim_tfidf + 0.7 * sim_bert
            
            # Ensure similarities are valid numbers
            if np.isnan(sim_tfidf) or np.isinf(sim_tfidf):
                sim_tfidf = 0.0
            if np.isnan(sim_bert) or np.isinf(sim_bert):
                sim_bert = 0.0
            if np.isnan(sim_combined) or np.isinf(sim_combined):
                sim_combined = 0.0
            
            # Store detailed similarity information
            sims.append((j, sim_combined, sim_tfidf, sim_bert))
        except (json.JSONDecodeError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Could not parse vectors for journal {j.name}: {e}")
            continue

    sims.sort(key=lambda x: x[1], reverse=True)
    ranked = sims[:top_k]

    # audit trail
    q = QueryRun(query_text=abstract, model_used="ensemble")
    db.add(q)
    
    try:
        db.commit()
        
        for rank, (j, score, _, _) in enumerate(ranked, 1):
            # Ensure score is a valid float before inserting
            if np.isnan(score) or np.isinf(score):
                score = 0.0
            db.add(Recommendation(query_id=q.id, journal_id=j.id,
                                  similarity=float(score), rank=rank))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
        # Continue with results even if audit fails
        pass

    # Create detailed results with all similarity scores
    results = []
    for j, sim_combined, sim_tfidf, sim_bert in ranked:
        result = {
            "journal_name": j.name,
            "display_name": j.display_name or j.name,
            "similarity_combined": round(sim_combined, 4),
            "similarity_tfidf": round(sim_tfidf, 4),
            "similarity_bert": round(sim_bert, 4),
            "impact_factor": j.impact_factor or 0.0,
            "is_open_access": j.is_open_access,
            "publisher": j.publisher or "Unknown",
            "issn": j.issn,
            "eissn": j.eissn,
            "subjects": json.loads(j.subjects) if j.subjects and j.subjects.strip() else []
        }
        results.append(result)
    db.close()
    return results


def get_ranking_comparisons(abstract: str, top_k: int = settings.TOP_K):
    """Get rankings by different criteria for comparison"""
    db = SessionLocal()
    
    # Get similarity-based ranking (reuse existing logic)
    similarity_results = rank_journals(abstract, top_k)
    
    # Get impact factor ranking
    journals_with_impact = db.query(Journal).filter(
        Journal.impact_factor.isnot(None),
        Journal.impact_factor > 0
    ).order_by(Journal.impact_factor.desc()).limit(top_k).all()
    
    impact_results = []
    for j in journals_with_impact:
        impact_results.append({
            "journal_name": j.name,
            "display_name": j.display_name or j.name,
            "impact_factor": j.impact_factor,
            "is_open_access": j.is_open_access,
            "publisher": j.publisher or "Unknown",
            "subjects": []  # Simplified for now
        })
    
    # Get TF-IDF only ranking
    tfidf_results = rank_by_tfidf_only(abstract, top_k, db)
    
    # Get BERT only ranking  
    bert_results = rank_by_bert_only(abstract, top_k, db)
    
    db.close()
    return {
        "similarity_ranking": similarity_results,
        "tfidf_only_ranking": tfidf_results,
        "bert_only_ranking": bert_results,
        "impact_factor_ranking": impact_results
    }


def rank_by_tfidf_only(abstract: str, top_k: int, db=None):
    """Rank journals using only TF-IDF similarity"""
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    # encode query
    vec_tfidf_sparse = tfidf.transform([abstract])
    vec_tfidf = np.array(vec_tfidf_sparse.todense()).flatten()

    journals = db.query(Journal).join(JournalProfile).all()
    sims = []
    
    for j in journals:
        p = j.profile
        if not p or not p.tfidf_vector:
            continue
            
        try:
            v_tfidf = np.array(json.loads(p.tfidf_vector))
            
            def cosine_sim(a, b):
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return np.dot(a, b) / (norm_a * norm_b)
            
            sim_tfidf = cosine_sim(vec_tfidf, v_tfidf)
            
            if np.isnan(sim_tfidf) or np.isinf(sim_tfidf):
                sim_tfidf = 0.0
            
            sims.append((j, sim_tfidf))
        except (json.JSONDecodeError, ValueError) as e:
            continue

    sims.sort(key=lambda x: x[1], reverse=True)
    ranked = sims[:top_k]

    results = []
    for j, score in ranked:
        results.append({
            "journal_name": j.name,
            "display_name": j.display_name or j.name,
            "similarity_tfidf": round(score, 4),
            "impact_factor": j.impact_factor or 0.0,
            "is_open_access": j.is_open_access,
            "publisher": j.publisher or "Unknown"
        })
    
    if close_db:
        db.close()
    
    return results


def rank_by_bert_only(abstract: str, top_k: int, db=None):
    """Rank journals using only BERT similarity"""
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    # encode query
    vec_bert = bert.encode([abstract])[0]

    journals = db.query(Journal).join(JournalProfile).all()
    sims = []
    
    for j in journals:
        p = j.profile
        if not p or not p.bert_vector:
            continue
            
        try:
            v_bert = np.array(json.loads(p.bert_vector))
            
            def cosine_sim(a, b):
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return np.dot(a, b) / (norm_a * norm_b)
            
            sim_bert = cosine_sim(vec_bert, v_bert)
            
            if np.isnan(sim_bert) or np.isinf(sim_bert):
                sim_bert = 0.0
            
            sims.append((j, sim_bert))
        except (json.JSONDecodeError, ValueError) as e:
            continue

    sims.sort(key=lambda x: x[1], reverse=True)
    ranked = sims[:top_k]

    results = []
    for j, score in ranked:
        results.append({
            "journal_name": j.name,
            "display_name": j.display_name or j.name,
            "similarity_bert": round(score, 4),
            "impact_factor": j.impact_factor or 0.0,
            "is_open_access": j.is_open_access,
            "publisher": j.publisher or "Unknown"
        })
    
    if close_db:
        db.close()
    
    return results


def analyze_text_distribution(abstract: str):
    """Analyze word distribution and frequency for visualization"""
    import re
    from collections import Counter
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    # Clean and tokenize text
    words = re.findall(r'\b[a-zA-Z]{3,}\b', abstract.lower())
    
    # Remove stop words
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    # Get word frequency
    word_freq = Counter(words)
    
    # Get TF-IDF and BERT representations
    vec_tfidf_sparse = tfidf.transform([abstract])
    vec_tfidf = np.array(vec_tfidf_sparse.todense()).flatten()
    vec_bert = bert.encode([abstract])
    
    return {
        "word_frequency": dict(word_freq.most_common(20)),
        "total_words": len(words),
        "unique_words": len(set(words)),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "sentence_count": len(re.split(r'[.!?]+', abstract)),
        "tfidf_vector_stats": {
            "dimensions": len(vec_tfidf),
            "non_zero_features": np.count_nonzero(vec_tfidf),
            "max_value": float(np.max(vec_tfidf)),
            "mean_value": float(np.mean(vec_tfidf))
        },
        "bert_vector_stats": {
            "dimensions": len(vec_bert),
            "max_value": float(np.max(vec_bert)),
            "min_value": float(np.min(vec_bert)),
            "mean_value": float(np.mean(vec_bert)),
            "std_value": float(np.std(vec_bert))
        }
    }
