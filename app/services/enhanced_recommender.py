#!/usr/bin/env python3
"""
Enhanced Journal Recommendation System
Significantly improved accuracy through better matching strategies
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from app.models.base import SessionLocal
from app.models.entities import Journal, JournalProfile
from typing import List, Tuple
import re

class EnhancedRecommender:
    """Enhanced recommender with multiple matching strategies"""
    
    def __init__(self):
        # Load models
        self.bert = SentenceTransformer("all-MiniLM-L6-v2")
        self.tfidf = TfidfVectorizer(max_features=20_000, stop_words="english", ngram_range=(1, 2))
        
        # Enhanced journal representations
        self.journal_data = {}
        self.subject_keywords = {}
        self._load_journal_data()
        
    def _load_journal_data(self):
        """Load and enhance journal data with better representations"""
        
        db = SessionLocal()
        try:
            journals = db.query(Journal).all()
            
            # Create enhanced journal representations
            corpus = []
            for journal in journals:
                # Enhanced text representation
                enhanced_text = self._create_enhanced_journal_text(journal)
                corpus.append(enhanced_text)
                
                # Store journal data
                self.journal_data[journal.id] = {
                    'journal': journal,
                    'enhanced_text': enhanced_text,
                    'keywords': self._extract_keywords(journal),
                    'subjects': self._parse_subjects(journal.subjects)
                }
            
            # Fit TF-IDF on enhanced corpus
            if corpus:
                self.tfidf.fit(corpus)
                print(f"✅ Enhanced TF-IDF fitted on {len(corpus)} enhanced journal descriptions")
            
        finally:
            db.close()
    
    def _create_enhanced_journal_text(self, journal: Journal) -> str:
        """Create rich text representation of journal"""
        
        parts = []
        
        # Core journal info
        parts.append(journal.name)
        if journal.publisher:
            parts.append(journal.publisher)
        
        # Subject information (most important for matching)
        if journal.subjects:
            subjects = journal.subjects.replace(';', ' ').replace(',', ' ')
            parts.append(subjects)
        
        # Infer field from journal name and publisher
        inferred_fields = self._infer_field_keywords(journal)
        if inferred_fields:
            parts.extend(inferred_fields)
        
        return " ".join(parts).lower()
    
    def _infer_field_keywords(self, journal: Journal) -> List[str]:
        """Infer research field keywords from journal name and publisher"""
        
        text = f"{journal.name} {journal.publisher or ''}".lower()
        keywords = []
        
        # Computer Science & AI
        if any(term in text for term in ['computer', 'computing', 'artificial intelligence', 'machine learning', 
                                         'data science', 'software', 'algorithm', 'neural', 'AI']):
            keywords.extend(['computer science', 'artificial intelligence', 'machine learning', 'algorithms', 
                           'data science', 'software engineering', 'neural networks'])
        
        # Medicine & Health
        if any(term in text for term in ['medicine', 'medical', 'health', 'clinical', 'biomedical', 'journal of medicine']):
            keywords.extend(['medicine', 'medical research', 'clinical studies', 'healthcare', 'biomedical', 
                           'patient care', 'medical treatment', 'disease'])
        
        # Engineering
        if any(term in text for term in ['engineering', 'technology', 'materials', 'mechanical', 'electrical']):
            keywords.extend(['engineering', 'technology', 'materials science', 'mechanical engineering', 
                           'electrical engineering', 'innovation'])
        
        # Economics & Finance
        if any(term in text for term in ['economic', 'finance', 'business', 'management', 'market']):
            keywords.extend(['economics', 'finance', 'business', 'management', 'financial analysis', 
                           'market research', 'economic policy'])
        
        # Environmental Science
        if any(term in text for term in ['environment', 'climate', 'ecology', 'sustainability', 'green']):
            keywords.extend(['environmental science', 'climate change', 'ecology', 'sustainability', 
                           'conservation', 'renewable energy'])
        
        # Biology & Life Sciences
        if any(term in text for term in ['biology', 'life sciences', 'genetics', 'molecular', 'cell']):
            keywords.extend(['biology', 'life sciences', 'genetics', 'molecular biology', 'cell biology', 
                           'biotechnology'])
        
        return keywords
    
    def _extract_keywords(self, journal: Journal) -> List[str]:
        """Extract key terms from journal name and subjects"""
        
        text = f"{journal.name} {journal.subjects or ''}".lower()
        
        # Extract important terms
        keywords = []
        important_terms = re.findall(r'\b[a-z]{4,}\b', text)  # Words 4+ chars
        keywords.extend(important_terms[:10])  # Top 10 terms
        
        return keywords
    
    def _parse_subjects(self, subjects_str: str) -> List[str]:
        """Parse and clean subject strings"""
        
        if not subjects_str:
            return []
        
        # Split on common separators and clean
        subjects = re.split(r'[;,]', subjects_str)
        cleaned = [s.strip().lower() for s in subjects if s.strip()]
        return cleaned[:5]  # Top 5 subjects
    
    def enhanced_rank_journals(self, abstract: str, top_k: int = 10) -> List[Tuple]:
        """Enhanced ranking with multiple strategies"""
        
        # Prepare abstract
        abstract_clean = self._preprocess_abstract(abstract)
        
        # Get vectors
        abstract_tfidf = self.tfidf.transform([abstract_clean])
        abstract_bert = self.bert.encode([abstract_clean])[0]
        
        # Score all journals
        scores = []
        for journal_id, data in self.journal_data.items():
            
            # 1. Semantic similarity (BERT)
            journal_bert = self.bert.encode([data['enhanced_text']])[0]
            semantic_score = np.dot(abstract_bert, journal_bert) / (
                np.linalg.norm(abstract_bert) * np.linalg.norm(journal_bert))
            
            # 2. Keyword similarity (TF-IDF)
            journal_tfidf = self.tfidf.transform([data['enhanced_text']])
            keyword_score = cosine_similarity(abstract_tfidf, journal_tfidf)[0][0]
            
            # 3. Subject/field matching
            field_score = self._calculate_field_score(abstract_clean, data)
            
            # 4. Term overlap score
            overlap_score = self._calculate_term_overlap(abstract_clean, data)
            
            # Combined score with enhanced weighting
            final_score = (
                0.40 * semantic_score +     # BERT semantic understanding
                0.25 * keyword_score +      # TF-IDF keyword matching  
                0.20 * field_score +        # Field/subject alignment
                0.15 * overlap_score        # Direct term overlap
            )
            
            scores.append((
                data['journal'], 
                final_score, 
                semantic_score, 
                keyword_score, 
                field_score, 
                overlap_score
            ))
        
        # Sort and return top results
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _preprocess_abstract(self, abstract: str) -> str:
        """Clean and enhance abstract for better matching"""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', abstract.strip().lower())
        
        # Remove common academic phrases that don't help matching
        stop_phrases = [
            'this paper', 'this study', 'we present', 'we propose', 'we demonstrate',
            'in this work', 'our results', 'our findings', 'we show', 'we found'
        ]
        
        for phrase in stop_phrases:
            text = text.replace(phrase, '')
        
        return text
    
    def _calculate_field_score(self, abstract: str, journal_data: dict) -> float:
        """Calculate field-specific matching score"""
        
        # Extract field indicators from abstract
        abstract_fields = []
        
        # Check for field indicators
        if any(term in abstract for term in ['machine learning', 'neural network', 'algorithm', 'computer', 'AI']):
            abstract_fields.append('computer science')
        
        if any(term in abstract for term in ['patient', 'clinical', 'medical', 'disease', 'treatment', 'health']):
            abstract_fields.append('medicine')
        
        if any(term in abstract for term in ['material', 'engineering', 'mechanical', 'design', 'technology']):
            abstract_fields.append('engineering')
        
        if any(term in abstract for term in ['economic', 'financial', 'market', 'business', 'cost']):
            abstract_fields.append('economics')
        
        if any(term in abstract for term in ['environment', 'climate', 'ecology', 'sustainability', 'carbon']):
            abstract_fields.append('environmental')
        
        # Score based on field overlap
        journal_subjects = journal_data.get('subjects', [])
        journal_keywords = journal_data.get('keywords', [])
        
        score = 0.0
        for field in abstract_fields:
            if any(field in subject for subject in journal_subjects):
                score += 1.0
            if any(field.replace(' ', '') in keyword for keyword in journal_keywords):
                score += 0.5
        
        return min(score / max(len(abstract_fields), 1), 1.0)
    
    def _calculate_term_overlap(self, abstract: str, journal_data: dict) -> float:
        """Calculate direct term overlap score"""
        
        # Extract important terms from abstract
        abstract_terms = set(re.findall(r'\b[a-z]{4,}\b', abstract))
        
        # Get journal terms
        journal_text = journal_data['enhanced_text']
        journal_terms = set(re.findall(r'\b[a-z]{4,}\b', journal_text))
        
        # Calculate overlap
        if not abstract_terms or not journal_terms:
            return 0.0
        
        overlap = len(abstract_terms.intersection(journal_terms))
        return overlap / len(abstract_terms.union(journal_terms))

# Global instance
enhanced_recommender = None

def get_enhanced_recommender():
    """Get or create enhanced recommender instance"""
    global enhanced_recommender
    if enhanced_recommender is None:
        enhanced_recommender = EnhancedRecommender()
    return enhanced_recommender

def enhanced_rank_journals(abstract: str, top_k: int = 10):
    """Enhanced ranking function for API"""
    
    recommender = get_enhanced_recommender()
    results = recommender.enhanced_rank_journals(abstract, top_k)
    
    # Format for API response
    recommendations = []
    for i, (journal, final_score, semantic, keyword, field, overlap) in enumerate(results, 1):
        recommendations.append({
            'journal_name': journal.name,
            'similarity_score': float(final_score),
            'rank': i,
            'detailed_scores': {
                'semantic': float(semantic),
                'keyword': float(keyword), 
                'field': float(field),
                'overlap': float(overlap)
            }
        })
    
    return recommendations