import json, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings
from app.models.base import SessionLocal
from app.models.entities import Journal, JournalProfile, QueryRun, Recommendation
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# load models once
bert_general = SentenceTransformer("all-MiniLM-L6-v2")  # General purpose, 384 dimensions
# Note: SciBERT removed for now - database vectors are 384-dim only
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
    db = SessionLocal()

    # Extract keywords from abstract
    abstract_keywords = extract_keywords(abstract, top_n=10)
    
    # encode query (removed SciBERT - vectors are 384-dim only)
    vec_tfidf_sparse = tfidf.transform([abstract])
    vec_tfidf = np.array(vec_tfidf_sparse.todense()).flatten()  # Convert sparse to dense properly
    vec_bert_general = bert_general.encode([abstract])[0]
    
    # Pre-encode abstract for title similarity (encode once instead of 353 times!)
    vec_abstract_for_title = bert_general.encode([abstract[:200]])[0]  # Use first 200 chars for title matching

    # fetch candidates
    journals = db.query(Journal).join(JournalProfile).all()
    
    # Batch encode all journal titles at once (MUCH faster than one-by-one)
    journal_names = [j.name for j in journals if j.profile and j.profile.tfidf_vector and j.profile.bert_vector]
    journal_title_vectors = bert_general.encode(journal_names) if journal_names else []
    
    sims = []
    title_vec_idx = 0
    
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
            
            # Component 1 & 2: General BERT similarity (stored vectors are 384-dim general BERT)
            sim_bert_general = cosine_sim(vec_bert_general, v_bert)
            sim_bert_scientific = sim_bert_general  # Same as general for now (no SciBERT vectors stored)
            
            # Component 3: TF-IDF similarity
            sim_tfidf = cosine_sim(vec_tfidf, v_tfidf)
            
            # Component 4: Title similarity (using pre-encoded vectors)
            sim_title = cosine_sim(vec_abstract_for_title, journal_title_vectors[title_vec_idx])
            title_vec_idx += 1
            
            # Component 5: Keyword similarity
            journal_text = f"{j.name} {j.display_name or ''} {j.publisher or ''}"
            sim_keyword = calculate_keyword_similarity(abstract_keywords, journal_text)
            
            # Component 6: Impact factor boost (normalized)
            impact_boost = normalize_impact_factor(j.impact_factor)
            
            # Component 7: Field matching boost
            journal_subjects = json.loads(j.subjects) if j.subjects and j.subjects.strip() else []
            field_boost = calculate_field_matching(abstract_keywords, journal_subjects)
            
            # SIMPLIFIED COMBINED SCORE (SciBERT removed until we rebuild vectors with 768-dim)
            # Using higher weight for general BERT since it's the only real BERT score
            sim_combined = (
                0.50 * sim_bert_general +    # Doubled from 0.25 (no SciBERT)
                0.20 * sim_tfidf +
                0.10 * sim_title +
                0.10 * sim_keyword +
                0.05 * impact_boost +
                0.05 * field_boost
            )
            
            # Ensure similarities are valid numbers
            if np.isnan(sim_tfidf) or np.isinf(sim_tfidf):
                sim_tfidf = 0.0
            if np.isnan(sim_bert_general) or np.isinf(sim_bert_general):
                sim_bert_general = 0.0
            if np.isnan(sim_bert_scientific) or np.isinf(sim_bert_scientific):
                sim_bert_scientific = 0.0
            if np.isnan(sim_combined) or np.isinf(sim_combined):
                sim_combined = 0.0
            
            # Store detailed similarity information
            sims.append((j, sim_combined, sim_tfidf, sim_bert_general, sim_bert_scientific, 
                        sim_title, sim_keyword, impact_boost, field_boost))
        except (json.JSONDecodeError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Could not parse vectors for journal {j.name}: {e}")
            continue

    sims.sort(key=lambda x: x[1], reverse=True)
    ranked = sims[:top_k]

    # audit trail
    q = QueryRun(query_text=abstract, model_used="advanced_ensemble")
    db.add(q)
    
    try:
        db.commit()
        
        for rank, (j, score, _, _, _, _, _, _, _) in enumerate(ranked, 1):
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

    # Create detailed results with all similarity scores and components
    results = []
    for j, sim_combined, sim_tfidf, sim_bert_gen, sim_bert_sci, sim_title, sim_keyword, impact_boost, field_boost in ranked:
        result = {
            "journal_name": j.name,
            "display_name": j.display_name or j.name,
            "similarity_combined": round(float(sim_combined), 4),
            "similarity_tfidf": round(float(sim_tfidf), 4),
            "similarity_bert": round(float((sim_bert_gen + sim_bert_sci) / 2), 4),  # Average BERT for backward compatibility
            "similarity_bert_general": round(float(sim_bert_gen), 4),
            "similarity_bert_scientific": round(float(sim_bert_sci), 4),
            "similarity_title": round(float(sim_title), 4),
            "similarity_keyword": round(float(sim_keyword), 4),
            "impact_factor_boost": round(float(impact_boost), 4),
            "field_matching_boost": round(float(field_boost), 4),
            "impact_factor": float(j.impact_factor) if j.impact_factor else 0.0,
            "is_open_access": bool(j.is_open_access),
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
    vec_bert = bert_general.encode([abstract])  # Use general BERT
    
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
