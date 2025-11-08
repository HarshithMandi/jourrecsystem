"""
User Metrics Module

Tracks and analyzes user behavior including:
- Query patterns and trends
- Popular journals and topics
- User interaction patterns
- Recommendation acceptance rates
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.models.base import SessionLocal
from app.models.entities import QueryRun, Recommendation, Journal
from sqlalchemy import func, and_


class UserMetrics:
    """Track and analyze user behavior metrics"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def get_query_patterns(self, hours: int = 24) -> Dict:
        """
        Analyze query patterns and trends
        
        Returns:
            - Query frequency over time
            - Peak usage hours
            - Query length distribution
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        queries = self.db.query(QueryRun).filter(
            QueryRun.timestamp >= cutoff
        ).all()
        
        if not queries:
            return {'error': 'No queries in time window'}
        
        # Analyze query timing
        hourly_distribution = defaultdict(int)
        daily_distribution = defaultdict(int)
        query_lengths = []
        
        for query in queries:
            hour = query.timestamp.hour
            day = query.timestamp.strftime('%Y-%m-%d')
            hourly_distribution[hour] += 1
            daily_distribution[day] += 1
            query_lengths.append(len(query.query_text))
        
        # Find peak hours
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1]) if hourly_distribution else (0, 0)
        
        return {
            'total_queries': len(queries),
            'time_window_hours': hours,
            'hourly_distribution': dict(sorted(hourly_distribution.items())),
            'daily_distribution': dict(sorted(daily_distribution.items())),
            'peak_hour': {
                'hour': peak_hour[0],
                'query_count': peak_hour[1]
            },
            'query_length_stats': {
                'avg_length': float(np.mean(query_lengths)),
                'median_length': float(np.median(query_lengths)),
                'min_length': min(query_lengths),
                'max_length': max(query_lengths)
            },
            'queries_per_day': round(len(queries) / (hours / 24), 2) if hours >= 24 else None
        }
    
    def get_popular_journals(self, limit: int = 20, hours: int = 168) -> Dict:
        """
        Get most frequently recommended journals
        
        Args:
            limit: Number of top journals to return
            hours: Time window (default 1 week)
        
        Returns:
            - Top recommended journals
            - Recommendation frequency
            - Average similarity scores
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Get journal recommendation counts
        results = self.db.query(
            Journal.id,
            Journal.name,
            Journal.publisher,
            Journal.is_open_access,
            func.count(Recommendation.id).label('recommendation_count'),
            func.avg(Recommendation.similarity).label('avg_similarity')
        ).join(
            Recommendation, Journal.id == Recommendation.journal_id
        ).join(
            QueryRun, Recommendation.query_id == QueryRun.id
        ).filter(
            QueryRun.timestamp >= cutoff
        ).group_by(
            Journal.id, Journal.name, Journal.publisher, Journal.is_open_access
        ).order_by(
            func.count(Recommendation.id).desc()
        ).limit(limit).all()
        
        popular_journals = []
        for result in results:
            popular_journals.append({
                'journal_name': result.name,
                'publisher': result.publisher or 'Unknown',
                'is_open_access': result.is_open_access,
                'recommendation_count': result.recommendation_count,
                'avg_similarity_score': round(float(result.avg_similarity), 3)
            })
        
        return {
            'top_journals': popular_journals,
            'time_window_hours': hours,
            'total_unique_journals_recommended': len(popular_journals)
        }
    
    def get_topic_trends(self, hours: int = 168) -> Dict:
        """
        Analyze trending topics in user queries
        
        Returns:
            - Most common keywords
            - Subject area distribution
            - Emerging topics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        queries = self.db.query(QueryRun).filter(
            QueryRun.timestamp >= cutoff
        ).all()
        
        if not queries:
            return {'error': 'No queries in time window'}
        
        # Extract keywords from all queries
        all_words = []
        for query in queries:
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', query.query_text.lower())
            # Remove common stop words
            stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 
                         'their', 'which', 'would', 'about', 'could', 'other'}
            words = [w for w in words if w not in stop_words]
            all_words.extend(words)
        
        # Count word frequency
        word_freq = Counter(all_words)
        
        # Get recommended journals and their subjects
        recommendations = self.db.query(
            Journal.subjects
        ).join(
            Recommendation, Journal.id == Recommendation.journal_id
        ).join(
            QueryRun, Recommendation.query_id == QueryRun.id
        ).filter(
            QueryRun.timestamp >= cutoff,
            Journal.subjects.isnot(None)
        ).all()
        
        # Analyze subject distribution
        subject_counter = Counter()
        for rec in recommendations:
            try:
                subjects = json.loads(rec.subjects)
                if isinstance(subjects, list):
                    for subj in subjects:
                        if isinstance(subj, dict) and 'display_name' in subj:
                            subject_counter[subj['display_name']] += 1
            except:
                continue
        
        return {
            'top_keywords': dict(word_freq.most_common(30)),
            'top_subjects': dict(subject_counter.most_common(20)),
            'total_queries_analyzed': len(queries),
            'time_window_hours': hours
        }
    
    def get_user_interaction_patterns(self) -> Dict:
        """
        Analyze how users interact with the system
        
        Returns:
            - Session patterns
            - Average recommendations per query
            - Return user patterns
        """
        # Get all queries
        all_queries = self.db.query(QueryRun).all()
        
        if not all_queries:
            return {'error': 'No query data available'}
        
        # Group by session
        session_groups = defaultdict(list)
        for query in all_queries:
            session_groups[query.session_id].append(query)
        
        # Analyze sessions
        queries_per_session = [len(queries) for queries in session_groups.values()]
        
        # Analyze recommendations per query
        recommendations_per_query = []
        for query in all_queries:
            rec_count = self.db.query(Recommendation).filter(
                Recommendation.query_id == query.id
            ).count()
            recommendations_per_query.append(rec_count)
        
        return {
            'total_sessions': len(session_groups),
            'total_queries': len(all_queries),
            'session_stats': {
                'avg_queries_per_session': round(float(np.mean(queries_per_session)), 2),
                'median_queries_per_session': float(np.median(queries_per_session)),
                'max_queries_per_session': max(queries_per_session),
                'single_query_sessions': sum(1 for q in queries_per_session if q == 1)
            },
            'recommendation_stats': {
                'avg_recommendations_per_query': round(float(np.mean(recommendations_per_query)), 2),
                'median_recommendations_per_query': float(np.median(recommendations_per_query))
            },
            'user_retention': {
                'multi_query_sessions': len([s for s in queries_per_session if s > 1]),
                'multi_query_session_percentage': round(
                    len([s for s in queries_per_session if s > 1]) / len(session_groups) * 100, 2
                ) if session_groups else 0
            }
        }
    
    def get_recommendation_acceptance_rate(self, hours: int = 168) -> Dict:
        """
        Estimate recommendation acceptance/clickthrough rates
        
        Note: This requires tracking user clicks/selections, which needs to be
        implemented in the API endpoints
        
        Returns:
            - Placeholder structure for implementation
        """
        return {
            'note': 'Acceptance rate tracking requires logging user selections',
            'implementation_steps': [
                '1. Add click/selection tracking to API endpoints',
                '2. Store selected recommendations in separate table',
                '3. Calculate acceptance_rate = selected / shown',
                '4. Track position-based acceptance (rank 1 vs rank 10)'
            ],
            'metrics_to_track': {
                'overall_acceptance_rate': 'Percentage of recommendations clicked',
                'position_based_acceptance': 'Click rate by ranking position',
                'time_to_first_click': 'How quickly users find relevant journal',
                'journals_clicked_per_query': 'Average number of journals selected'
            }
        }
    
    def get_query_refinement_patterns(self) -> Dict:
        """
        Analyze how users refine their queries
        
        Returns:
            - Query modification patterns
            - Refinement strategies
        """
        # Group queries by session
        sessions = self.db.query(QueryRun.session_id).distinct().all()
        
        refinement_count = 0
        total_sessions = 0
        
        for session_tuple in sessions:
            session_id = session_tuple[0]
            queries = self.db.query(QueryRun).filter(
                QueryRun.session_id == session_id
            ).order_by(QueryRun.timestamp).all()
            
            if len(queries) > 1:
                refinement_count += len(queries) - 1
                total_sessions += 1
        
        return {
            'total_sessions_with_refinements': total_sessions,
            'total_refinements': refinement_count,
            'avg_refinements_per_session': round(refinement_count / total_sessions, 2) if total_sessions > 0 else 0,
            'note': 'Advanced refinement analysis requires query similarity comparison'
        }
    
    def get_open_access_preference(self, hours: int = 168) -> Dict:
        """
        Analyze user preference for open access journals
        
        Returns:
            - Open access selection rate
            - Comparison with availability
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Get all recommendations
        total_recommendations = self.db.query(Recommendation).join(QueryRun).filter(
            QueryRun.timestamp >= cutoff
        ).count()
        
        # Get open access recommendations
        oa_recommendations = self.db.query(Recommendation).join(
            Journal, Recommendation.journal_id == Journal.id
        ).join(QueryRun).filter(
            QueryRun.timestamp >= cutoff,
            Journal.is_open_access == True
        ).count()
        
        # Get overall OA availability
        total_journals = self.db.query(Journal).count()
        oa_journals = self.db.query(Journal).filter(Journal.is_open_access == True).count()
        
        return {
            'open_access_recommendation_rate': round(oa_recommendations / total_recommendations * 100, 2) if total_recommendations > 0 else 0,
            'open_access_availability_rate': round(oa_journals / total_journals * 100, 2) if total_journals > 0 else 0,
            'oa_recommended': oa_recommendations,
            'total_recommended': total_recommendations,
            'time_window_hours': hours
        }
    
    def export_metrics(self, filepath: str):
        """Export user metrics to JSON file"""
        metrics = {
            'query_patterns': self.get_query_patterns(hours=168),
            'popular_journals': self.get_popular_journals(),
            'topic_trends': self.get_topic_trends(),
            'interaction_patterns': self.get_user_interaction_patterns(),
            'open_access_preference': self.get_open_access_preference(),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db'):
            self.db.close()
