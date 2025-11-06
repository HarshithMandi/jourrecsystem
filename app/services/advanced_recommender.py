#!/usr/bin/env python3
"""
Advanced Multi-Strategy Recommendation System
Uses multiple sophisticated strategies to improve accuracy
"""

import json
import numpy as np
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from app.models.base import SessionLocal
from app.models.entities import Journal, Work

class AdvancedRecommender:
    """Advanced recommender with multiple sophisticated strategies"""
    
    def __init__(self):
        # Enhanced models with different specializations
        self.bert_general = SentenceTransformer("all-MiniLM-L6-v2")  # General purpose
        self.bert_scientific = SentenceTransformer("allenai/scibert_scivocab_uncased")  # Scientific texts
        
        # Multiple TF-IDF models for different aspects
        self.tfidf_content = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 3))
        self.tfidf_keywords = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        
        # Enhanced data structures
        self.journal_profiles = {}
        self.field_mappings = {}
        self.publisher_mappings = {}
        
        self._build_advanced_profiles()
        
    def _build_advanced_profiles(self):
        """Build comprehensive journal profiles"""
        
        db = SessionLocal()
        try:
            journals = db.query(Journal).all()
            works = db.query(Work).all()
            
            print(f"📊 Building advanced profiles for {len(journals)} journals and {len(works)} papers...")
            
            # Group papers by journal
            papers_by_journal = defaultdict(list)
            for work in works:
                if work.abstract and work.journal_id:
                    papers_by_journal[work.journal_id].append(work.abstract)
            
            # Create comprehensive journal profiles
            all_journal_texts = []
            all_keyword_texts = []
            
            for journal in journals:
                profile = self._create_advanced_profile(journal, papers_by_journal.get(journal.id, []))
                self.journal_profiles[journal.id] = profile
                
                all_journal_texts.append(profile['full_text'])
                all_keyword_texts.append(profile['keyword_text'])
            
            # Fit TF-IDF models on all journal data
            if all_journal_texts:
                self.tfidf_content.fit(all_journal_texts)
                self.tfidf_keywords.fit(all_keyword_texts)
                print("✅ Advanced TF-IDF models fitted")
            
            print(f"✅ Built {len(self.journal_profiles)} advanced journal profiles")
            
        finally:
            db.close()
    
    def _create_advanced_profile(self, journal: Journal, paper_abstracts: List[str]) -> Dict:
        """Create comprehensive journal profile"""
        
        # Basic info
        name = journal.name or ""
        publisher = journal.publisher or ""
        subjects = journal.subjects or ""
        
        # Enhanced field classification
        field_keywords = self._classify_journal_field(journal)
        
        # Abstract analysis (if available)
        abstract_insights = self._analyze_abstracts(paper_abstracts) if paper_abstracts else {}
        
        # Create different text representations
        full_text = f"{name} {publisher} {subjects} {' '.join(field_keywords)}"
        if abstract_insights.get('common_terms'):
            full_text += f" {' '.join(abstract_insights['common_terms'][:20])}"
        
        keyword_text = f"{' '.join(field_keywords)} {subjects}"
        
        return {
            'journal': journal,
            'full_text': full_text.lower(),
            'keyword_text': keyword_text.lower(),
            'field_keywords': field_keywords,
            'subjects_list': self._parse_subjects(subjects),
            'publisher': publisher.lower(),
            'abstract_insights': abstract_insights,
            'research_areas': self._identify_research_areas(journal, field_keywords)
        }
    
    def _classify_journal_field(self, journal: Journal) -> List[str]:
        """Advanced field classification"""
        
        text = f"{journal.name} {journal.publisher or ''} {journal.subjects or ''}".lower()
        keywords = []
        
        # Computer Science & AI
        cs_terms = ['computer', 'computing', 'artificial intelligence', 'machine learning', 'data science', 
                   'software', 'algorithm', 'neural', 'AI', 'programming', 'information technology',
                   'cybersecurity', 'blockchain', 'quantum computing', 'robotics', 'automation']
        if any(term in text for term in cs_terms):
            keywords.extend(['computer science', 'artificial intelligence', 'machine learning', 'algorithms',
                           'data science', 'software engineering', 'information technology', 'computational methods'])
        
        # Medicine & Health
        med_terms = ['medicine', 'medical', 'health', 'clinical', 'biomedical', 'patient', 'disease',
                    'therapy', 'treatment', 'diagnosis', 'healthcare', 'pharmaceutical', 'nursing',
                    'surgery', 'oncology', 'cardiology', 'neurology', 'psychiatry']
        if any(term in text for term in med_terms):
            keywords.extend(['medicine', 'medical research', 'clinical studies', 'healthcare',
                           'biomedical research', 'patient care', 'disease treatment', 'medical diagnosis'])
        
        # Engineering & Technology
        eng_terms = ['engineering', 'technology', 'materials', 'mechanical', 'electrical', 'civil',
                    'chemical', 'aerospace', 'manufacturing', 'design', 'innovation', 'systems']
        if any(term in text for term in eng_terms):
            keywords.extend(['engineering', 'technology', 'materials science', 'mechanical engineering',
                           'electrical engineering', 'innovation', 'technological development'])
        
        # Life Sciences & Biology
        bio_terms = ['biology', 'life sciences', 'genetics', 'molecular', 'cell', 'biochemistry',
                    'microbiology', 'ecology', 'evolution', 'genomics', 'proteomics', 'biotechnology']
        if any(term in text for term in bio_terms):
            keywords.extend(['biology', 'life sciences', 'genetics', 'molecular biology',
                           'cell biology', 'biotechnology', 'biological research'])
        
        # Physics & Chemistry
        phys_terms = ['physics', 'chemistry', 'physical', 'chemical', 'quantum', 'atomic',
                     'molecular dynamics', 'thermodynamics', 'optics', 'spectroscopy']
        if any(term in text for term in phys_terms):
            keywords.extend(['physics', 'chemistry', 'physical sciences', 'chemical research',
                           'materials physics', 'theoretical physics'])
        
        # Economics & Business
        econ_terms = ['economic', 'economics', 'finance', 'business', 'management', 'market',
                     'financial', 'accounting', 'banking', 'investment', 'trade', 'commerce']
        if any(term in text for term in econ_terms):
            keywords.extend(['economics', 'finance', 'business', 'management', 'financial analysis',
                           'market research', 'economic policy', 'business strategy'])
        
        # Environmental Science
        env_terms = ['environment', 'environmental', 'climate', 'ecology', 'sustainability',
                    'conservation', 'renewable', 'green', 'carbon', 'pollution', 'biodiversity']
        if any(term in text for term in env_terms):
            keywords.extend(['environmental science', 'climate change', 'ecology', 'sustainability',
                           'conservation', 'renewable energy', 'environmental protection'])
        
        return list(set(keywords))  # Remove duplicates
    
    def _analyze_abstracts(self, abstracts: List[str]) -> Dict:
        """Analyze abstracts to extract common themes"""
        
        if not abstracts:
            return {}
        
        # Combine all abstracts
        combined_text = " ".join(abstracts).lower()
        
        # Extract common terms
        words = re.findall(r'\b[a-z]{4,}\b', combined_text)
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Get most common meaningful terms
        common_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
        
        return {
            'common_terms': [term for term, freq in common_terms if freq > 1],
            'total_abstracts': len(abstracts),
            'avg_length': sum(len(abs) for abs in abstracts) / len(abstracts)
        }
    
    def _parse_subjects(self, subjects_str: str) -> List[str]:
        """Parse subject string into clean list"""
        
        if not subjects_str:
            return []
        
        subjects = re.split(r'[;,]', subjects_str.lower())
        return [s.strip() for s in subjects if s.strip()]
    
    def _identify_research_areas(self, journal: Journal, field_keywords: List[str]) -> List[str]:
        """Identify specific research areas"""
        
        text = f"{journal.name} {journal.subjects or ''}".lower()
        areas = []
        
        # Specific research area patterns
        area_patterns = {
            'machine learning': ['machine learning', 'neural network', 'deep learning', 'AI'],
            'natural language processing': ['natural language', 'NLP', 'text mining', 'linguistics'],
            'medical imaging': ['medical imaging', 'radiology', 'MRI', 'CT scan'],
            'cancer research': ['cancer', 'oncology', 'tumor', 'carcinoma'],
            'climate science': ['climate', 'global warming', 'atmosphere', 'weather'],
            'materials engineering': ['materials', 'polymers', 'composites', 'nanotechnology'],
            'financial modeling': ['financial', 'econometrics', 'risk', 'portfolio'],
            'biotechnology': ['biotech', 'genetic engineering', 'bioinformatics', 'genomics']
        }
        
        for area, patterns in area_patterns.items():
            if any(pattern in text for pattern in patterns):
                areas.append(area)
        
        return areas
    
    def advanced_rank_journals(self, abstract: str, top_k: int = 10) -> List[Tuple]:
        """Advanced ranking with multiple sophisticated strategies"""
        
        # Preprocess abstract
        abstract_clean = self._preprocess_text(abstract)
        
        # Multiple embeddings
        abstract_bert_general = self.bert_general.encode([abstract_clean])[0]
        try:
            abstract_bert_scientific = self.bert_scientific.encode([abstract_clean])[0]
        except:
            abstract_bert_scientific = abstract_bert_general  # Fallback
        
        # Multiple TF-IDF representations
        abstract_tfidf_content = self.tfidf_content.transform([abstract_clean])
        abstract_keywords = self._extract_research_keywords(abstract)
        abstract_tfidf_keywords = self.tfidf_keywords.transform([" ".join(abstract_keywords)])
        
        scores = []
        
        for journal_id, profile in self.journal_profiles.items():
            
            # 1. Semantic similarity (multiple BERT models)
            journal_bert_general = self.bert_general.encode([profile['full_text']])[0]
            semantic_general = self._safe_cosine_similarity(abstract_bert_general, journal_bert_general)
            
            try:
                journal_bert_scientific = self.bert_scientific.encode([profile['full_text']])[0]
                semantic_scientific = self._safe_cosine_similarity(abstract_bert_scientific, journal_bert_scientific)
            except:
                semantic_scientific = semantic_general
            
            # 2. Content similarity (TF-IDF)
            journal_tfidf_content = self.tfidf_content.transform([profile['full_text']])
            content_similarity = cosine_similarity(abstract_tfidf_content, journal_tfidf_content)[0][0]
            
            # 3. Keyword matching
            journal_tfidf_keywords = self.tfidf_keywords.transform([profile['keyword_text']])
            keyword_similarity = cosine_similarity(abstract_tfidf_keywords, journal_tfidf_keywords)[0][0]
            
            # 4. Field/subject alignment
            field_score = self._advanced_field_matching(abstract, abstract_keywords, profile)
            
            # 5. Research area matching
            area_score = self._research_area_matching(abstract, profile)
            
            # 6. Abstract insights matching (if available)
            insight_score = self._insight_matching(abstract_keywords, profile)
            
            # Advanced weighted combination
            final_score = (
                0.25 * semantic_general +        # General BERT
                0.20 * semantic_scientific +     # Scientific BERT
                0.20 * content_similarity +      # TF-IDF content
                0.15 * keyword_similarity +      # TF-IDF keywords
                0.10 * field_score +             # Field alignment
                0.05 * area_score +              # Research area
                0.05 * insight_score             # Abstract insights
            )
            
            # Apply impact factor boost for significant accuracy improvement
            final_score = self._apply_impact_factor_boost(final_score, profile['journal'])
            
            # Apply field matching boost for additional accuracy
            abstract_fields = self._classify_abstract_field(abstract)
            journal_fields = self._classify_journal_field(profile['journal'])
            final_score = self._apply_field_matching_boost(final_score, abstract_fields, journal_fields)
            
            scores.append((
                profile['journal'],
                final_score,
                {
                    'semantic_general': semantic_general,
                    'semantic_scientific': semantic_scientific,
                    'content': content_similarity,
                    'keywords': keyword_similarity,
                    'field': field_score,
                    'area': area_score,
                    'insights': insight_score
                }
            ))
        
        # Sort and return
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _safe_cosine_similarity(self, vec1, vec2):
        """Safe cosine similarity calculation"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip().lower())
        
        # Remove academic boilerplate
        stop_phrases = [
            'this paper', 'this study', 'we present', 'we propose', 'we demonstrate',
            'in this work', 'our results', 'our findings', 'we show', 'we found',
            'in conclusion', 'to conclude', 'in summary'
        ]
        
        for phrase in stop_phrases:
            text = text.replace(phrase, ' ')
        
        return re.sub(r'\s+', ' ', text).strip()
    
    def _extract_research_keywords(self, abstract: str) -> List[str]:
        """Extract important research keywords"""
        
        # Technical terms and methods
        keywords = []
        
        # Extract multi-word terms
        text = abstract.lower()
        
        # Common research patterns
        patterns = [
            r'machine learning', r'deep learning', r'neural network', r'artificial intelligence',
            r'natural language processing', r'computer vision', r'data science',
            r'clinical trial', r'randomized controlled', r'systematic review',
            r'finite element', r'computational fluid', r'molecular dynamics',
            r'gene expression', r'protein folding', r'cell culture',
            r'climate change', r'renewable energy', r'sustainability',
            r'financial model', r'risk assessment', r'economic analysis'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                keywords.append(pattern.replace(' ', '_'))
        
        # Extract single important terms
        important_terms = re.findall(r'\b(?:algorithm|method|model|analysis|approach|technique|framework|system|dataset|benchmark|evaluation|optimization|classification|regression|clustering|prediction|forecasting|treatment|therapy|diagnosis|intervention|protocol|mechanism|pathway|process|structure|function|performance|efficiency|accuracy|precision|recall|sensitivity|specificity)\b', text)
        keywords.extend(important_terms)
        
        return list(set(keywords))[:15]  # Top 15 unique keywords
    
    def _advanced_field_matching(self, abstract: str, keywords: List[str], profile: Dict) -> float:
        """Advanced field matching with multiple strategies"""
        
        score = 0.0
        
        # Match with journal field keywords
        journal_keywords = set(profile['field_keywords'])
        abstract_text = abstract.lower()
        
        # Direct keyword matching
        for keyword in journal_keywords:
            if keyword in abstract_text:
                score += 1.0
        
        # Research area matching
        research_areas = set(profile['research_areas'])
        for area in research_areas:
            if area.replace('_', ' ') in abstract_text:
                score += 0.5
        
        # Subject matching
        subjects = set(profile['subjects_list'])
        for subject in subjects:
            if subject in abstract_text:
                score += 0.3
        
        return min(score / max(len(journal_keywords), 1), 1.0)
    
    def _research_area_matching(self, abstract: str, profile: Dict) -> float:
        """Match specific research areas"""
        
        research_areas = profile.get('research_areas', [])
        if not research_areas:
            return 0.0
        
        abstract_lower = abstract.lower()
        matches = 0
        
        for area in research_areas:
            if area.replace('_', ' ') in abstract_lower:
                matches += 1
        
        return matches / len(research_areas)
    
    def _insight_matching(self, abstract_keywords: List[str], profile: Dict) -> float:
        """Match against journal's abstract insights"""
        
        insights = profile.get('abstract_insights', {})
        common_terms = insights.get('common_terms', [])
        
        if not common_terms:
            return 0.0
        
        abstract_terms = set(abstract_keywords)
        journal_terms = set(common_terms[:20])  # Top 20 terms
        
        if not abstract_terms or not journal_terms:
            return 0.0
        
        overlap = len(abstract_terms.intersection(journal_terms))
        return overlap / len(abstract_terms.union(journal_terms))
    
    def _apply_impact_factor_boost(self, base_score: float, journal) -> float:
        """Apply impact factor boost for high-impact journals"""
        
        if not hasattr(journal, 'impact_factor') or not journal.impact_factor:
            return base_score
        
        try:
            impact_factor = float(journal.impact_factor)
            
            # Logarithmic boost based on impact factor tiers
            if impact_factor >= 10.0:       # Nature, Science, Cell (top tier)
                boost = 0.25                # +25% boost for elite journals
            elif impact_factor >= 5.0:      # High impact journals
                boost = 0.20                # +20% boost
            elif impact_factor >= 3.0:      # Above average impact
                boost = 0.15                # +15% boost
            elif impact_factor >= 2.0:      # Good impact journals
                boost = 0.10                # +10% boost
            elif impact_factor >= 1.0:      # Indexed journals
                boost = 0.05                # +5% boost
            else:
                boost = 0.02                # Small boost for any impact factor
            
            # Apply boost with cap at 1.0
            boosted_score = base_score * (1.0 + boost)
            
            # Extra boost for very high scores + high impact (compound effect)
            if base_score > 0.3 and impact_factor >= 5.0:
                boosted_score *= 1.05       # Additional 5% for strong matches with high-impact journals
            
            return min(boosted_score, 1.0)
            
        except (ValueError, TypeError):
            return base_score
    
    def _classify_journal_field(self, journal) -> List[str]:
        """Classify journal into research fields for better matching"""
        
        text = f"{journal.name} {journal.publisher or ''} {journal.subjects or ''}".lower()
        fields = []
        
        # Field classification patterns
        field_patterns = {
            'artificial_intelligence': [
                'artificial intelligence', 'machine learning', 'neural network', 'deep learning',
                'AI', 'ML', 'computer intelligence', 'pattern recognition', 'data mining'
            ],
            'medical_research': [
                'medicine', 'medical', 'clinical', 'health', 'biomedical', 'journal of medicine',
                'oncology', 'cardiology', 'neurology', 'radiology', 'surgery', 'therapy'
            ],
            'computer_science': [
                'computer science', 'computing', 'software', 'algorithm', 'programming',
                'information technology', 'cybersecurity', 'database', 'network'
            ],
            'engineering': [
                'engineering', 'technology', 'materials', 'mechanical', 'electrical', 'civil',
                'aerospace', 'industrial', 'chemical engineering', 'robotics'
            ],
            'life_sciences': [
                'biology', 'biochemistry', 'genetics', 'molecular', 'cell', 'microbiology',
                'biotechnology', 'bioinformatics', 'ecology', 'evolution'
            ],
            'physics_chemistry': [
                'physics', 'chemistry', 'physical', 'chemical', 'quantum', 'materials science',
                'crystallography', 'spectroscopy', 'analytical chemistry'
            ],
            'economics_finance': [
                'economics', 'economic', 'finance', 'financial', 'business', 'management',
                'accounting', 'marketing', 'operations research'
            ],
            'environmental_science': [
                'environment', 'environmental', 'climate', 'ecology', 'sustainability',
                'conservation', 'renewable energy', 'pollution', 'green technology'
            ],
            'social_sciences': [
                'psychology', 'sociology', 'anthropology', 'political science', 'education',
                'linguistics', 'communication', 'social work'
            ],
            'mathematics_statistics': [
                'mathematics', 'mathematical', 'statistics', 'statistical', 'probability',
                'optimization', 'numerical analysis', 'applied mathematics'
            ]
        }
        
        # Classify based on patterns
        for field, patterns in field_patterns.items():
            if any(pattern in text for pattern in patterns):
                fields.append(field)
        
        # Handle high-impact general journals
        if any(term in text for term in ['nature', 'science', 'cell', 'pnas']):
            fields.append('high_impact_general')
        
        return fields if fields else ['general']
    
    def _classify_abstract_field(self, abstract: str) -> List[str]:
        """Classify abstract into research fields"""
        
        text = abstract.lower()
        fields = []
        
        # Field indicators in abstracts
        field_indicators = {
            'artificial_intelligence': [
                'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
                'AI', 'ML', 'classification', 'regression', 'clustering', 'natural language processing',
                'computer vision', 'reinforcement learning', 'supervised learning'
            ],
            'medical_research': [
                'patient', 'clinical', 'disease', 'treatment', 'therapy', 'medical', 'health',
                'diagnosis', 'clinical trial', 'healthcare', 'medicine', 'surgical', 'therapeutic'
            ],
            'computer_science': [
                'algorithm', 'software', 'programming', 'database', 'network', 'system',
                'computational', 'data structure', 'cybersecurity', 'blockchain'
            ],
            'engineering': [
                'design', 'manufacturing', 'materials', 'mechanical', 'electrical', 'optimization',
                'simulation', 'modeling', 'engineering', 'control system', 'robotics'
            ],
            'life_sciences': [
                'gene', 'protein', 'cell', 'molecular', 'genetic', 'biological', 'organism',
                'enzyme', 'DNA', 'RNA', 'biochemical', 'microbial', 'ecological'
            ],
            'physics_chemistry': [
                'quantum', 'molecular dynamics', 'spectroscopy', 'crystallographic', 'chemical',
                'physical', 'atomic', 'particle', 'thermodynamic', 'kinetic'
            ],
            'economics_finance': [
                'economic', 'financial', 'market', 'investment', 'risk', 'portfolio', 'cost',
                'price', 'economic model', 'financial analysis', 'business'
            ],
            'environmental_science': [
                'climate', 'environment', 'ecological', 'sustainability', 'conservation',
                'renewable energy', 'carbon', 'greenhouse', 'biodiversity', 'ecosystem'
            ],
            'social_sciences': [
                'social', 'psychological', 'behavior', 'education', 'learning', 'cognitive',
                'survey', 'interview', 'qualitative', 'quantitative research'
            ],
            'mathematics_statistics': [
                'mathematical', 'statistical', 'probability', 'theorem', 'proof', 'equation',
                'optimization', 'numerical', 'stochastic', 'regression analysis'
            ]
        }
        
        # Classify based on content
        for field, indicators in field_indicators.items():
            if any(indicator in text for indicator in indicators):
                fields.append(field)
        
        return fields if fields else ['general']
    
    def _apply_field_matching_boost(self, base_score: float, abstract_fields: List[str], journal_fields: List[str]) -> float:
        """Apply boost for field alignment between abstract and journal"""
        
        if not abstract_fields or not journal_fields:
            return base_score
        
        # Calculate field overlap
        abstract_set = set(abstract_fields)
        journal_set = set(journal_fields)
        
        # Direct field matches
        direct_matches = len(abstract_set.intersection(journal_set))
        
        # Related field matches (e.g., AI and computer science)
        related_field_groups = [
            {'artificial_intelligence', 'computer_science', 'mathematics_statistics'},
            {'medical_research', 'life_sciences'},
            {'physics_chemistry', 'engineering'},
            {'economics_finance', 'mathematics_statistics'},
            {'environmental_science', 'life_sciences'}
        ]
        
        related_matches = 0
        for group in related_field_groups:
            if abstract_set.intersection(group) and journal_set.intersection(group):
                related_matches += 0.5
        
        # Calculate boost
        total_overlap = direct_matches + related_matches
        max_possible = max(len(abstract_fields), len(journal_fields))
        
        if max_possible > 0:
            field_alignment = total_overlap / max_possible
            
            # Apply boost
            if field_alignment >= 0.8:
                boost = 0.20        # Strong field match
            elif field_alignment >= 0.5:
                boost = 0.15        # Good field match
            elif field_alignment >= 0.3:
                boost = 0.10        # Moderate field match
            else:
                boost = 0.05        # Weak field match
            
            return min(base_score * (1.0 + boost), 1.0)
        
        return base_score

# Global instance
advanced_recommender = None

def get_advanced_recommender():
    """Get or create advanced recommender"""
    global advanced_recommender
    if advanced_recommender is None:
        advanced_recommender = AdvancedRecommender()
    return advanced_recommender

def advanced_rank_journals(abstract: str, top_k: int = 10):
    """Advanced ranking function for API"""
    
    recommender = get_advanced_recommender()
    results = recommender.advanced_rank_journals(abstract, top_k)
    
    # Format for API
    recommendations = []
    for i, (journal, final_score, detailed_scores) in enumerate(results, 1):
        recommendations.append({
            'journal_name': journal.name,
            'similarity_score': float(final_score),
            'rank': i,
            'detailed_scores': {k: float(v) for k, v in detailed_scores.items()}
        })
    
    return recommendations