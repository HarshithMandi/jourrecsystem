"""
Accuracy Metrics Module

Tracks and analyzes recommendation accuracy including:
- Similarity score distributions
- Ranking quality metrics
- Component contribution analysis
- Model performance comparison
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.models.base import SessionLocal
from app.models.entities import QueryRun, Recommendation, Journal
from sqlalchemy import func, and_


class AccuracyMetrics:
    """Track and analyze recommendation accuracy metrics"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def get_similarity_score_distribution(self, hours: int = 24) -> Dict:
        """
        Analyze distribution of similarity scores
        
        Returns:
            - Score distribution across bins
            - Mean, median, std of scores
            - High/low score counts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        recommendations = self.db.query(Recommendation).join(QueryRun).filter(
            QueryRun.timestamp >= cutoff
        ).all()
        
        if not recommendations:
            return {'error': 'No recommendations in time window'}
        
        scores = [r.similarity for r in recommendations]
        
        # Create distribution bins
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(scores, bins=bins)
        
        distribution = {}
        for i in range(len(bins)-1):
            label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
            distribution[label] = int(hist[i])
        
        return {
            'distribution': distribution,
            'statistics': {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75))
            },
            'high_quality_recommendations': len([s for s in scores if s > 0.7]),
            'medium_quality_recommendations': len([s for s in scores if 0.4 <= s <= 0.7]),
            'low_quality_recommendations': len([s for s in scores if s < 0.4]),
            'total_recommendations': len(scores),
            'time_window_hours': hours
        }
    
    def get_ranking_quality_metrics(self, hours: int = 24) -> Dict:
        """
        Analyze quality of rankings
        
        Returns:
            - Top-K accuracy metrics
            - Score gap analysis (difference between ranks)
            - Ranking diversity
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Get all queries in time window
        queries = self.db.query(QueryRun).filter(
            QueryRun.timestamp >= cutoff
        ).all()
        
        if not queries:
            return {'error': 'No queries in time window'}
        
        ranking_metrics = {
            'avg_score_drop': [],  # Score drop between consecutive ranks
            'top1_scores': [],
            'top3_scores': [],
            'top5_scores': [],
            'score_gaps': []
        }
        
        for query in queries:
            recommendations = self.db.query(Recommendation).filter(
                Recommendation.query_id == query.id
            ).order_by(Recommendation.rank).all()
            
            if not recommendations:
                continue
            
            scores = [r.similarity for r in recommendations]
            
            # Track top-K scores
            if len(scores) >= 1:
                ranking_metrics['top1_scores'].append(scores[0])
            if len(scores) >= 3:
                ranking_metrics['top3_scores'].extend(scores[:3])
            if len(scores) >= 5:
                ranking_metrics['top5_scores'].extend(scores[:5])
            
            # Calculate score drops between consecutive ranks
            if len(scores) >= 2:
                drops = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
                ranking_metrics['avg_score_drop'].extend(drops)
                
                # Score gap between rank 1 and rank 10 (if exists)
                if len(scores) >= 10:
                    gap = scores[0] - scores[9]
                    ranking_metrics['score_gaps'].append(gap)
        
        return {
            'top1_avg_score': float(np.mean(ranking_metrics['top1_scores'])) if ranking_metrics['top1_scores'] else 0,
            'top3_avg_score': float(np.mean(ranking_metrics['top3_scores'])) if ranking_metrics['top3_scores'] else 0,
            'top5_avg_score': float(np.mean(ranking_metrics['top5_scores'])) if ranking_metrics['top5_scores'] else 0,
            'avg_score_drop_between_ranks': float(np.mean(ranking_metrics['avg_score_drop'])) if ranking_metrics['avg_score_drop'] else 0,
            'avg_top1_to_top10_gap': float(np.mean(ranking_metrics['score_gaps'])) if ranking_metrics['score_gaps'] else 0,
            'ranking_consistency': self._calculate_ranking_consistency(ranking_metrics['avg_score_drop']),
            'total_queries_analyzed': len(queries)
        }
    
    def _calculate_ranking_consistency(self, score_drops: List[float]) -> float:
        """
        Calculate ranking consistency (lower std of score drops = more consistent)
        
        Returns value between 0-1 (higher = more consistent)
        """
        if not score_drops:
            return 0.0
        
        std = np.std(score_drops)
        # Normalize: low std = high consistency
        # Assuming std rarely exceeds 0.1 in good rankings
        consistency = max(0, 1 - (std / 0.1))
        return round(float(consistency), 3)
    
    def get_component_contribution_analysis(self) -> Dict:
        """
        Analyze contribution of different similarity components
        
        This requires storing component scores, which would need to be added
        to the Recommendation model or tracked separately
        """
        # Placeholder for component analysis
        # In production, you'd track: TF-IDF, BERT, title, keyword, impact factor, field matching
        
        return {
            'note': 'Component contributions need to be logged separately',
            'components': [
                'bert_similarity',
                'tfidf_similarity', 
                'title_similarity',
                'keyword_similarity',
                'impact_factor_boost',
                'field_matching_boost'
            ]
        }
    
    def get_model_performance_comparison(self) -> Dict:
        """
        Compare different ranking models/methods
        
        Returns:
            - Comparison of TF-IDF only vs BERT only vs Combined
            - Score distributions per model
        """
        # This would require logging which model was used
        # For now, return structure for implementation
        
        return {
            'note': 'Model comparison requires separate tracking of model variants',
            'models_to_compare': [
                'tfidf_only',
                'bert_only',
                'hybrid_combined',
                'impact_factor_weighted'
            ]
        }
    
    def get_recommendation_diversity(self, hours: int = 24) -> Dict:
        """
        Analyze diversity of recommendations
        
        Returns:
            - Number of unique journals recommended
            - Distribution across publishers
            - Open access vs traditional ratio
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Get all recommended journals
        recommended_journal_ids = self.db.query(Recommendation.journal_id).join(QueryRun).filter(
            QueryRun.timestamp >= cutoff
        ).distinct().all()
        
        journal_ids = [j[0] for j in recommended_journal_ids]
        
        if not journal_ids:
            return {'error': 'No recommendations in time window'}
        
        journals = self.db.query(Journal).filter(Journal.id.in_(journal_ids)).all()
        
        # Analyze publishers
        publishers = [j.publisher for j in journals if j.publisher]
        publisher_dist = defaultdict(int)
        for p in publishers:
            publisher_dist[p] += 1
        
        # Open access analysis
        open_access_count = sum(1 for j in journals if j.is_open_access)
        
        # Subject diversity
        all_subjects = []
        for j in journals:
            if j.subjects:
                try:
                    subjects = json.loads(j.subjects)
                    if isinstance(subjects, list):
                        for subj in subjects:
                            if isinstance(subj, dict) and 'display_name' in subj:
                                all_subjects.append(subj['display_name'])
                except:
                    pass
        
        subject_dist = defaultdict(int)
        for subj in all_subjects:
            subject_dist[subj] += 1
        
        return {
            'unique_journals_recommended': len(journal_ids),
            'unique_publishers': len(publisher_dist),
            'top_publishers': dict(sorted(publisher_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
            'open_access_ratio': round(open_access_count / len(journals), 3) if journals else 0,
            'open_access_count': open_access_count,
            'traditional_count': len(journals) - open_access_count,
            'subject_diversity': len(subject_dist),
            'top_subjects': dict(sorted(subject_dist.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def get_precision_at_k(self, k: int = 10) -> Dict:
        """
        Calculate Precision@K metrics
        
        Note: This requires ground truth relevance labels, which typically
        come from user feedback or expert annotations
        """
        return {
            'note': f'Precision@{k} requires ground truth relevance labels',
            'implementation': 'Track user clicks/selections on recommendations',
            'formula': 'Precision@K = (Relevant items in top-K) / K'
        }
    
    def get_ndcg_score(self) -> Dict:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)
        
        Standard metric for ranking quality
        """
        return {
            'note': 'NDCG requires ground truth relevance scores',
            'implementation': 'Compare predicted ranking with ideal ranking',
            'formula': 'NDCG = DCG / IDCG'
        }
    
    def get_mean_reciprocal_rank(self) -> Dict:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Measures how quickly relevant results appear in rankings
        """
        return {
            'note': 'MRR requires identifying first relevant result per query',
            'implementation': 'Track position of first clicked/selected recommendation',
            'formula': 'MRR = average(1/rank_of_first_relevant_item)'
        }
    
    def export_metrics(self, filepath: str):
        """Export accuracy metrics to JSON file"""
        metrics = {
            'similarity_distribution': self.get_similarity_score_distribution(),
            'ranking_quality': self.get_ranking_quality_metrics(),
            'recommendation_diversity': self.get_recommendation_diversity(),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db'):
            self.db.close()
