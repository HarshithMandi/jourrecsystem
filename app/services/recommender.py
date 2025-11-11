import json, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings
from app.models.base import SessionLocal
from app.models.entities import Journal, JournalProfile, QueryRun, Recommendation
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
import time

# load models once
bert_general = SentenceTransformer("all-MiniLM-L6-v2")  # General purpose, 384 dimensions
# Note: SciBERT removed for now - database vectors are 384-dim only
tfidf = TfidfVectorizer(max_features=20_000, stop_words="english")

# CACHE: Pre-load all journal vectors into memory for faster access
_journal_cache = None
_cache_timestamp = None
CACHE_TTL = 3600  # 1 hour cache

def get_journal_cache():
    """Get cached journal data or reload if stale"""
    global _journal_cache, _cache_timestamp
    
    current_time = time.time()
    if _journal_cache is None or _cache_timestamp is None or (current_time - _cache_timestamp) > CACHE_TTL:
        print("Loading journal cache...")
        _session = SessionLocal()
        try:
            journals = _session.query(Journal).join(JournalProfile).all()
            
            _journal_cache = {
                'journals': [],
                'tfidf_vectors': [],
                'bert_vectors': [],
                'names': [],
                'metadata': []
            }
            
            for j in journals:
                p = j.profile
                if not p or not p.tfidf_vector or not p.bert_vector:
                    continue
                
                try:
                    v_tfidf = np.array(json.loads(p.tfidf_vector))
                    v_bert = np.array(json.loads(p.bert_vector))
                    
                    _journal_cache['journals'].append(j)
                    _journal_cache['tfidf_vectors'].append(v_tfidf)
                    _journal_cache['bert_vectors'].append(v_bert)
                    _journal_cache['names'].append(j.name)
                    _journal_cache['metadata'].append({
                        'id': j.id,
                        'name': j.name,
                        'display_name': j.display_name,
                        'publisher': j.publisher,
                        'impact_factor': j.impact_factor,
                        'is_open_access': j.is_open_access,
                        'issn': j.issn,
                        'eissn': j.eissn,
                        'subjects': json.loads(j.subjects) if j.subjects and j.subjects.strip() else []
                    })
                except:
                    continue
            
            # Convert to numpy arrays for vectorized operations
            _journal_cache['tfidf_matrix'] = np.array(_journal_cache['tfidf_vectors'])
            _journal_cache['bert_matrix'] = np.array(_journal_cache['bert_vectors'])
            
            _cache_timestamp = current_time
            print(f"✓ Loaded {len(_journal_cache['journals'])} journals into cache")
        finally:
            _session.close()
    
    return _journal_cache

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

# Pre-load cache on startup
get_journal_cache()

def extract_keywords(text: str, top_n: int = 10):
    """Extract top keywords from text using TF-IDF"""
    vec_tfidf_sparse = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    vec_tfidf = vec_tfidf_sparse.toarray()[0]
    
    # Get top N keywords
    top_indices = vec_tfidf.argsort()[-top_n:][::-1]
    keywords = [feature_names[i] for i in top_indices if vec_tfidf[i] > 0]
    return keywords

def calculate_keyword_similarity(abstract_keywords, journal_text):
    """Calculate keyword overlap between abstract and journal"""
    journal_text_lower = journal_text.lower()
    matches = sum(1 for keyword in abstract_keywords if keyword.lower() in journal_text_lower)
    if len(abstract_keywords) == 0:
        return 0.0
    return matches / len(abstract_keywords)

def calculate_title_similarity(vec_abstract_encoded, journal_name: str):
    """Calculate similarity between abstract and journal title using pre-encoded abstract vector"""
    try:
        # Encode journal title only (abstract is already encoded)
        vec_title = bert_general.encode([journal_name])[0]
        
        norm_a = np.linalg.norm(vec_abstract_encoded)
        norm_b = np.linalg.norm(vec_title)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_abstract_encoded, vec_title) / (norm_a * norm_b)
    except:
        return 0.0

def normalize_impact_factor(impact_factor, max_impact=100.0):
    """Normalize impact factor to 0-1 range"""
    if impact_factor is None or impact_factor <= 0:
        return 0.0
    return min(impact_factor / max_impact, 1.0)

def calculate_field_matching(abstract_keywords, journal_subjects):
    """Calculate field/subject matching score"""
    if not journal_subjects or len(abstract_keywords) == 0:
        return 0.0
    
    # Extract display names from subject dicts (subjects is a list of dicts)
    try:
        if isinstance(journal_subjects, list) and len(journal_subjects) > 0:
            if isinstance(journal_subjects[0], dict):
                # Extract display_name from dict objects
                subject_names = [subj.get('display_name', '') for subj in journal_subjects if subj.get('display_name')]
            else:
                # Assume they're strings
                subject_names = [str(subj) for subj in journal_subjects]
        else:
            return 0.0
    except:
        return 0.0
    
    if not subject_names:
        return 0.0
    
    # Convert both to lowercase for matching
    abstract_kw_lower = [kw.lower() for kw in abstract_keywords]
    journal_subj_lower = [subj.lower() for subj in subject_names]
    
    # Check for keyword-subject matches
    matches = 0
    for kw in abstract_kw_lower:
        for subj in journal_subj_lower:
            if kw in subj or subj in kw:
                matches += 1
                break
    
    return min(matches / len(abstract_keywords), 1.0)

def rank_journals(abstract: str, top_k: int = settings.TOP_K):
    """
    Optimized journal ranking using cached vectors and vectorized operations
    """
    db = SessionLocal()
    
    # Get cached journal data
    cache = get_journal_cache()
    
    if not cache or len(cache['journals']) == 0:
        db.close()
        return []
    
    # Extract keywords from abstract
    abstract_keywords = extract_keywords(abstract, top_n=10)
    
    # Encode query ONCE
    vec_tfidf_sparse = tfidf.transform([abstract])
    vec_tfidf = np.array(vec_tfidf_sparse.todense()).flatten()
    vec_bert_general = bert_general.encode([abstract])[0]
    
    # VECTORIZED OPERATIONS - Much faster than looping!
    # Normalize vectors for cosine similarity
    vec_tfidf_norm = vec_tfidf / (np.linalg.norm(vec_tfidf) + 1e-8)
    vec_bert_norm = vec_bert_general / (np.linalg.norm(vec_bert_general) + 1e-8)
    
    # Normalize all journal vectors at once
    tfidf_matrix_norm = cache['tfidf_matrix'] / (np.linalg.norm(cache['tfidf_matrix'], axis=1, keepdims=True) + 1e-8)
    bert_matrix_norm = cache['bert_matrix'] / (np.linalg.norm(cache['bert_matrix'], axis=1, keepdims=True) + 1e-8)
    
    # Calculate all similarities at once using matrix multiplication
    sim_tfidf_all = np.dot(tfidf_matrix_norm, vec_tfidf_norm)
    sim_bert_all = np.dot(bert_matrix_norm, vec_bert_norm)
    
    # Batch encode titles for title similarity (one batch operation instead of N individual ones)
    abstract_title = abstract[:200]
    vec_abstract_title = bert_general.encode([abstract_title])[0]
    vec_abstract_title_norm = vec_abstract_title / (np.linalg.norm(vec_abstract_title) + 1e-8)
    
    title_vectors = bert_general.encode(cache['names'])
    title_vectors_norm = title_vectors / (np.linalg.norm(title_vectors, axis=1, keepdims=True) + 1e-8)
    sim_title_all = np.dot(title_vectors_norm, vec_abstract_title_norm)
    
    # Calculate other components for each journal
    results = []
    for idx, (j, metadata) in enumerate(zip(cache['journals'], cache['metadata'])):
        sim_tfidf = float(sim_tfidf_all[idx])
        sim_bert = float(sim_bert_all[idx])
        sim_title = float(sim_title_all[idx])
        
        # Keyword similarity (still needs text processing)
        journal_text = f"{metadata['name']} {metadata['display_name'] or ''} {metadata['publisher'] or ''}"
        sim_keyword = calculate_keyword_similarity(abstract_keywords, journal_text)
        
        # Impact factor boost
        impact_boost = normalize_impact_factor(metadata['impact_factor'])
        
        # Field matching
        field_boost = calculate_field_matching(abstract_keywords, metadata['subjects'])
        
        # Combined score with optimized weights
        sim_combined = (
            0.50 * sim_bert +
            0.20 * sim_tfidf +
            0.10 * sim_title +
            0.10 * sim_keyword +
            0.05 * impact_boost +
            0.05 * field_boost
        )
        
        results.append({
            'journal': j,
            'metadata': metadata,
            'sim_combined': sim_combined,
            'sim_tfidf': sim_tfidf,
            'sim_bert': sim_bert,
            'sim_title': sim_title,
            'sim_keyword': sim_keyword,
            'impact_boost': impact_boost,
            'field_boost': field_boost
        })
    
    # Sort by combined similarity
    results.sort(key=lambda x: x['sim_combined'], reverse=True)
    ranked = results[:top_k]
    
    # Audit trail (async would be even better, but this works)
    q = QueryRun(query_text=abstract, model_used="advanced_ensemble_v2")
    db.add(q)
    
    try:
        db.commit()
        
        for rank, result in enumerate(ranked, 1):
            db.add(Recommendation(
                query_id=q.id,
                journal_id=result['metadata']['id'],
                similarity=float(result['sim_combined']),
                rank=rank
            ))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
    
    # Format output
    output = []
    for result in ranked:
        meta = result['metadata']
        output.append({
            "journal_name": meta['name'],
            "display_name": meta['display_name'] or meta['name'],
            "similarity_combined": round(float(result['sim_combined']), 4),
            "similarity_tfidf": round(float(result['sim_tfidf']), 4),
            "similarity_bert": round(float(result['sim_bert']), 4),
            "similarity_bert_general": round(float(result['sim_bert']), 4),
            "similarity_bert_scientific": round(float(result['sim_bert']), 4),
            "similarity_title": round(float(result['sim_title']), 4),
            "similarity_keyword": round(float(result['sim_keyword']), 4),
            "impact_factor_boost": round(float(result['impact_boost']), 4),
            "field_matching_boost": round(float(result['field_boost']), 4),
            "impact_factor": float(meta['impact_factor']) if meta['impact_factor'] else 0.0,
            "is_open_access": bool(meta['is_open_access']),
            "publisher": meta['publisher'] or "Unknown",
            "issn": meta['issn'],
            "eissn": meta['eissn'],
            "subjects": meta['subjects']
        })
    
    db.close()
    return output


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
            "impact_factor": float(j.impact_factor),
            "is_open_access": bool(j.is_open_access),
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
            "similarity_tfidf": round(float(score), 4),
            "impact_factor": float(j.impact_factor) if j.impact_factor else 0.0,
            "is_open_access": bool(j.is_open_access),
            "publisher": j.publisher or "Unknown"
        })
    
    if close_db:
        db.close()
    
    return results


def rank_by_bert_only(abstract: str, top_k: int, db=None):
    """Rank journals using only BERT similarity (using general BERT model)"""
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    # encode query using general BERT
    vec_bert = bert_general.encode([abstract])[0]

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
            "similarity_bert": round(float(score), 4),
            "impact_factor": float(j.impact_factor) if j.impact_factor else 0.0,
            "is_open_access": bool(j.is_open_access),
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
    vec_bert = bert_general.encode([abstract])[0]  # Extract first element from array
    
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
