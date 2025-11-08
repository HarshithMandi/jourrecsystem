"""
Ground Truth Validation System

Compares recommendation quality across different models:
- Hybrid (TF-IDF + BERT) 
- TF-IDF Only
- BERT Only

Evaluates using real publication data or expert-labeled test sets.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import json
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from app.models.base import SessionLocal
from app.models.entities import Journal, JournalProfile


class GroundTruthEvaluator:
    """
    Evaluates recommendation systems against ground truth data.
    
    Compares three approaches:
    1. Hybrid (current system)
    2. TF-IDF only
    3. BERT only
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.test_cases = []
        self.results = {
            'hybrid': {'top_10': [], 'top_20': []},
            'tfidf': {'top_10': [], 'top_20': []},
            'bert': {'top_10': [], 'top_20': []}
        }
        
    # ===== LOAD GROUND TRUTH DATA =====
    
    def create_synthetic_test_set(self, n_samples: int = 50) -> List[Dict]:
        """
        Create synthetic test set using actual papers from database OR synthetic abstracts.
        
        Strategy 1: Use actual published papers (TRUE ground truth)
        Strategy 2: If no papers, generate synthetic abstracts based on journal subjects
        """
        print(f"\n📝 Creating ground truth test set with {n_samples} samples...")
        
        # Try Strategy 1: Use actual published papers (TRUE ground truth)
        from app.models.entities import Work
        
        works = self.db.query(Work).filter(
            Work.abstract.isnot(None),
            Work.abstract != '',
            Work.journal_id.isnot(None)
        ).limit(n_samples * 2).all()
        
        test_cases = []
        
        if works:
            print("    Using actual published papers as queries (TRUE ground truth)...")
            for work in works:
                abstract = work.abstract
                if not abstract or len(str(abstract)) < 100:
                    continue
                
                journal = self.db.query(Journal).filter(Journal.id == work.journal_id).first()
                if not journal:
                    continue
                    
                test_case = {
                    'abstract': str(abstract)[:1000],
                    'true_journal_id': journal.id,
                    'true_journal_name': journal.name,
                    'metadata': {
                        'paper_title': work.title,
                        'publication_year': work.publication_year,
                        'publisher': journal.publisher
                    }
                }
                test_cases.append(test_case)
                
                if len(test_cases) >= n_samples:
                    break
        
        # Strategy 2: Generate synthetic abstracts based on journal subjects
        if len(test_cases) < n_samples:
            print("    Generating synthetic abstracts based on journal metadata...")
            
            # Get journals with rich metadata
            journals = self.db.query(Journal).filter(
                Journal.subjects.isnot(None),
                Journal.publisher.isnot(None)
            ).limit(n_samples * 2).all()
            
            # Templates for generating synthetic abstracts
            templates = [
                "This study investigates {topic} in {field}. We analyze {method} and present novel findings on {aspect}. Our results demonstrate significant advances in understanding {area}.",
                "Recent research in {field} has shown the importance of {topic}. This paper presents a comprehensive analysis of {method} with applications to {aspect}. We provide evidence for {area}.",
                "We examine {topic} using {method} in the context of {field}. Our findings contribute to the understanding of {aspect} and have implications for {area} research.",
                "This work focuses on {topic} and its role in {field}. Through {method}, we demonstrate new insights into {aspect} and advance knowledge in {area}.",
                "The {topic} phenomenon in {field} remains poorly understood. Using {method}, we investigate {aspect} and reveal important patterns in {area}."
            ]
            
            for journal in journals:
                if len(test_cases) >= n_samples:
                    break
                    
                # Parse subjects (stored as JSON string)
                try:
                    subjects = json.loads(journal.subjects) if journal.subjects else []
                    if isinstance(subjects, list) and subjects:
                        # Pick first subject as main field
                        field = subjects[0].get('display_name', 'science') if isinstance(subjects[0], dict) else str(subjects[0])
                    else:
                        field = "research"
                except:
                    field = "science"
                
                # Generate synthetic abstract using journal info
                import random
                template = random.choice(templates)
                
                # Generic terms that fit most research
                topics = ["methodology", "data analysis", "experimental design", "theoretical framework", "systematic review"]
                methods = ["quantitative analysis", "statistical modeling", "computational methods", "empirical investigation", "meta-analysis"]
                aspects = ["key variables", "underlying mechanisms", "critical factors", "emerging trends", "fundamental principles"]
                areas = ["current", "contemporary", "applied", "theoretical", "experimental"]
                
                abstract = template.format(
                    topic=random.choice(topics),
                    field=field.lower(),
                    method=random.choice(methods),
                    aspect=random.choice(aspects),
                    area=random.choice(areas)
                )
                
                test_case = {
                    'abstract': abstract,
                    'true_journal_id': journal.id,
                    'true_journal_name': journal.name,
                    'metadata': {
                        'synthetic': True,
                        'publisher': journal.publisher,
                        'subjects': journal.subjects,
                        'field': field
                    }
                }
                test_cases.append(test_case)
        
        self.test_cases = test_cases
        print(f"✓ Created {len(test_cases)} test cases")
        if any('synthetic' not in tc.get('metadata', {}) for tc in test_cases):
            print(f"   {sum(1 for tc in test_cases if 'synthetic' not in tc.get('metadata', {}))} from actual published papers (TRUE ground truth)")
        if any('synthetic' in tc.get('metadata', {}) for tc in test_cases):
            print(f"   {sum(1 for tc in test_cases if 'synthetic' in tc.get('metadata', {}))} synthetic cases based on journal metadata")
        return test_cases
    
    def load_from_json(self, filepath: str = 'metrics/test_data/targeted_test_cases.json') -> List[Dict]:
        """
        Load ground truth from JSON file.
        
        Expected format:
        [
            {
                "abstract": "Research abstract...",
                "true_journal_name": "Nature",
                "true_journal_id": 123
            },
            ...
        ]
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.test_cases = json.load(f)
            print(f"✓ Loaded {len(self.test_cases)} test cases from {filepath}")
            
            # Show test case info
            if self.test_cases and 'metadata' in self.test_cases[0]:
                if self.test_cases[0]['metadata'].get('targeted'):
                    print("   Using TARGETED test cases (high-quality matches expected)")
                elif self.test_cases[0]['metadata'].get('synthetic'):
                    print("   Using SYNTHETIC test cases")
            
            return self.test_cases
        except FileNotFoundError:
            print(f"✗ Test file not found: {filepath}")
            print("   Run 'python metrics/create_test_data.py' to generate targeted test data")
            return []
        except Exception as e:
            print(f"✗ Error loading ground truth: {e}")
            return []
    
    # ===== RECOMMENDATION METHODS =====
    
    def _get_hybrid_recommendations(self, abstract: str, top_k: int = 20) -> List[Tuple[int, str, float]]:
        """Get recommendations using current hybrid system."""
        from app.services import recommender
        
        results = recommender.rank_journals(abstract=abstract, top_k=top_k)
        
        # Look up journal IDs by name
        recommendations = []
        for r in results[:top_k]:
            journal = self.db.query(Journal).filter(Journal.name == r['journal_name']).first()
            if journal:
                recommendations.append((journal.id, r['journal_name'], r['similarity_combined']))
        
        return recommendations
    
    def _get_tfidf_recommendations(self, abstract: str, top_k: int = 20) -> List[Tuple[int, str, float]]:
        """Get recommendations using TF-IDF only."""
        from app.services import recommender
        
        results = recommender.rank_by_tfidf_only(abstract=abstract, top_k=top_k, db=self.db)
        
        # Look up journal IDs by name
        recommendations = []
        for r in results[:top_k]:
            journal = self.db.query(Journal).filter(Journal.name == r['journal_name']).first()
            if journal:
                recommendations.append((journal.id, r['journal_name'], r['similarity_tfidf']))
        
        return recommendations
    
    def _get_bert_recommendations(self, abstract: str, top_k: int = 20) -> List[Tuple[int, str, float]]:
        """Get recommendations using BERT only."""
        from app.services import recommender
        
        results = recommender.rank_by_bert_only(abstract=abstract, top_k=top_k, db=self.db)
        
        # Look up journal IDs by name
        recommendations = []
        for r in results[:top_k]:
            journal = self.db.query(Journal).filter(Journal.name == r['journal_name']).first()
            if journal:
                recommendations.append((journal.id, r['journal_name'], r['similarity_bert']))
        
        return recommendations
    
    # ===== EVALUATION METRICS =====
    
    def _calculate_hit_rate(self, recommendations: List[Tuple[int, str, float]], 
                           true_journal_id: int) -> bool:
        """Check if true journal appears in recommendations."""
        recommended_ids = [r[0] for r in recommendations]
        return true_journal_id in recommended_ids
    
    def _calculate_reciprocal_rank(self, recommendations: List[Tuple[int, str, float]], 
                                   true_journal_id: int) -> float:
        """Calculate reciprocal rank of true journal."""
        recommended_ids = [r[0] for r in recommendations]
        try:
            rank = recommended_ids.index(true_journal_id) + 1  # 1-indexed
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    def _calculate_rank(self, recommendations: List[Tuple[int, str, float]], 
                       true_journal_id: int) -> int:
        """Get rank of true journal (0 if not found)."""
        recommended_ids = [r[0] for r in recommendations]
        try:
            return recommended_ids.index(true_journal_id) + 1
        except ValueError:
            return 0
    
    # ===== MAIN EVALUATION =====
    
    def evaluate_all_models(self, top_k_values: List[int] = [10, 20]) -> Dict:
        """
        Evaluate all three models on test cases.
        
        Returns:
            Dictionary with results for each model and each K value
        """
        if not self.test_cases:
            print("⚠️  No test cases loaded. Attempting to load targeted test data...")
            self.load_from_json('metrics/test_data/targeted_test_cases.json')
            
        if not self.test_cases:
            print("⚠️  Creating synthetic test set...")
            self.create_synthetic_test_set(50)
        
        print(f"\n🔍 Evaluating {len(self.test_cases)} test cases...")
        print(f"   Models: Hybrid, TF-IDF Only, BERT Only")
        print(f"   Metrics: Hit Rate, MRR, Average Rank")
        print(f"   K values: {top_k_values}\n")
        
        results = {
            'hybrid': {f'top_{k}': {'hits': [], 'mrr': [], 'ranks': []} for k in top_k_values},
            'tfidf': {f'top_{k}': {'hits': [], 'mrr': [], 'ranks': []} for k in top_k_values},
            'bert': {f'top_{k}': {'hits': [], 'mrr': [], 'ranks': []} for k in top_k_values}
        }
        
        for i, test_case in enumerate(self.test_cases, 1):
            if i % 10 == 0:
                print(f"   Processing test case {i}/{len(self.test_cases)}...")
            
            abstract = test_case['abstract']
            true_journal_id = test_case['true_journal_id']
            
            # Get recommendations from all models
            max_k = max(top_k_values)
            
            try:
                hybrid_recs = self._get_hybrid_recommendations(abstract, max_k)
                tfidf_recs = self._get_tfidf_recommendations(abstract, max_k)
                bert_recs = self._get_bert_recommendations(abstract, max_k)
            except Exception as e:
                print(f"   ⚠️  Error processing test case {i}: {e}")
                continue
            
            # Evaluate for each K value
            for k in top_k_values:
                key = f'top_{k}'
                
                # Hybrid
                hybrid_k = hybrid_recs[:k]
                results['hybrid'][key]['hits'].append(
                    self._calculate_hit_rate(hybrid_k, true_journal_id)
                )
                results['hybrid'][key]['mrr'].append(
                    self._calculate_reciprocal_rank(hybrid_k, true_journal_id)
                )
                results['hybrid'][key]['ranks'].append(
                    self._calculate_rank(hybrid_k, true_journal_id)
                )
                
                # TF-IDF
                tfidf_k = tfidf_recs[:k]
                results['tfidf'][key]['hits'].append(
                    self._calculate_hit_rate(tfidf_k, true_journal_id)
                )
                results['tfidf'][key]['mrr'].append(
                    self._calculate_reciprocal_rank(tfidf_k, true_journal_id)
                )
                results['tfidf'][key]['ranks'].append(
                    self._calculate_rank(tfidf_k, true_journal_id)
                )
                
                # BERT
                bert_k = bert_recs[:k]
                results['bert'][key]['hits'].append(
                    self._calculate_hit_rate(bert_k, true_journal_id)
                )
                results['bert'][key]['mrr'].append(
                    self._calculate_reciprocal_rank(bert_k, true_journal_id)
                )
                results['bert'][key]['ranks'].append(
                    self._calculate_rank(bert_k, true_journal_id)
                )
        
        # Calculate summary statistics
        summary = {
            'hybrid': {},
            'tfidf': {},
            'bert': {}
        }
        
        for model in ['hybrid', 'tfidf', 'bert']:
            for k in top_k_values:
                key = f'top_{k}'
                hits = results[model][key]['hits']
                mrrs = results[model][key]['mrr']
                ranks = [r for r in results[model][key]['ranks'] if r > 0]
                
                summary[model][key] = {
                    'hit_rate': np.mean(hits) if hits else 0.0,
                    'mrr': np.mean(mrrs) if mrrs else 0.0,
                    'avg_rank': np.mean(ranks) if ranks else 0.0,
                    'n_samples': len(hits)
                }
        
        self.results = results
        self.summary = summary
        
        print("\n✓ Evaluation complete!\n")
        return summary
    
    def print_results(self):
        """Print evaluation results in a formatted table."""
        if not hasattr(self, 'summary'):
            print("⚠️  No results to display. Run evaluate_all_models() first.")
            return
        
        print("\n" + "="*80)
        print("GROUND TRUTH VALIDATION RESULTS")
        print("="*80)
        
        for k_value in ['top_10', 'top_20']:
            print(f"\n{'='*80}")
            print(f"  {k_value.replace('_', ' ').upper()} RESULTS")
            print(f"{'='*80}")
            print(f"{'Model':<20} {'Hit Rate':<15} {'MRR':<15} {'Avg Rank':<15}")
            print("-"*80)
            
            for model in ['hybrid', 'tfidf', 'bert']:
                model_name = {
                    'hybrid': 'Hybrid (TF-IDF+BERT)',
                    'tfidf': 'TF-IDF Only',
                    'bert': 'BERT Only'
                }[model]
                
                stats = self.summary[model][k_value]
                print(f"{model_name:<20} "
                      f"{stats['hit_rate']:<15.2%} "
                      f"{stats['mrr']:<15.4f} "
                      f"{stats['avg_rank']:<15.2f}")
        
        print("\n" + "="*80)
        print(f"Total test cases: {self.summary['hybrid']['top_10']['n_samples']}")
        print("="*80 + "\n")
    
    def export_results(self, output_dir: str = 'metrics/output'):
        """Export results to JSON file."""
        output_path = Path(output_dir) / 'ground_truth_results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'n_test_cases': len(self.test_cases),
            'summary': self.summary,
            'raw_results': {
                model: {
                    k: {
                        'hits': [int(h) for h in v['hits']],
                        'mrr': v['mrr'],
                        'ranks': v['ranks']
                    }
                    for k, v in model_results.items()
                }
                for model, model_results in self.results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Results exported to: {output_path}")


def main():
    """Run ground truth evaluation."""
    print("\n" + "="*80)
    print("GROUND TRUTH VALIDATION SYSTEM")
    print("="*80)
    print("Comparing: Hybrid (TF-IDF+BERT) vs TF-IDF Only vs BERT Only")
    print("Metrics: Hit Rate @ K=10,20 | MRR | Average Rank")
    print("="*80 + "\n")
    
    # Get database session
    db = SessionLocal()
    
    try:
        # Initialize evaluator
        evaluator = GroundTruthEvaluator(db)
        
        # Create test set (uses journal descriptions as queries)
        evaluator.create_synthetic_test_set(n_samples=50)
        
        # Evaluate all models
        summary = evaluator.evaluate_all_models(top_k_values=[10, 20])
        
        # Print results
        evaluator.print_results()
        
        # Export to JSON
        evaluator.export_results()
        
    finally:
        db.close()


if __name__ == '__main__':
    main()
