"""
System Metrics Module

Tracks and analyzes system health and data quality including:
- Database statistics
- Vector quality metrics
- Data coverage analysis
- System health indicators
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.models.base import SessionLocal
from app.models.entities import Journal, JournalProfile, Work, QueryRun, Recommendation
from sqlalchemy import func


class SystemMetrics:
    """Track and analyze system health and data quality metrics"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def get_database_statistics(self) -> Dict:
        """
        Comprehensive database statistics
        
        Returns:
            - Total counts of all entities
            - Growth metrics
            - Data completeness
        """
        stats = {
            'journals': {
                'total': self.db.query(Journal).count(),
                'with_profiles': self.db.query(Journal).join(JournalProfile).count(),
                'open_access': self.db.query(Journal).filter(Journal.is_open_access == True).count(),
                'with_impact_factor': self.db.query(Journal).filter(Journal.impact_factor.isnot(None)).count(),
                'with_issn': self.db.query(Journal).filter(Journal.issn.isnot(None)).count(),
                'with_subjects': self.db.query(Journal).filter(Journal.subjects.isnot(None)).count()
            },
            'profiles': {
                'total': self.db.query(JournalProfile).count(),
                'with_tfidf_vectors': self.db.query(JournalProfile).filter(
                    JournalProfile.tfidf_vector.isnot(None)
                ).count(),
                'with_bert_vectors': self.db.query(JournalProfile).filter(
                    JournalProfile.bert_vector.isnot(None)
                ).count(),
                'with_both_vectors': self.db.query(JournalProfile).filter(
                    JournalProfile.tfidf_vector.isnot(None),
                    JournalProfile.bert_vector.isnot(None)
                ).count()
            },
            'works': {
                'total': self.db.query(Work).count(),
                'with_abstracts': self.db.query(Work).filter(Work.abstract.isnot(None)).count()
            },
            'queries': {
                'total': self.db.query(QueryRun).count(),
                'with_recommendations': self.db.query(QueryRun).join(Recommendation).distinct().count()
            },
            'recommendations': {
                'total': self.db.query(Recommendation).count()
            }
        }
        
        # Calculate coverage percentages
        total_journals = stats['journals']['total']
        if total_journals > 0:
            stats['coverage'] = {
                'profile_coverage': round(stats['journals']['with_profiles'] / total_journals * 100, 2),
                'impact_factor_coverage': round(stats['journals']['with_impact_factor'] / total_journals * 100, 2),
                'subject_coverage': round(stats['journals']['with_subjects'] / total_journals * 100, 2),
                'vector_coverage': round(stats['profiles']['with_both_vectors'] / total_journals * 100, 2)
            }
        
        return stats
    
    def get_vector_quality_metrics(self) -> Dict:
        """
        Analyze quality of ML vectors
        
        Returns:
            - Vector dimension statistics
            - Non-zero feature counts
            - Vector norm distribution
            - Anomaly detection
        """
        profiles = self.db.query(JournalProfile).filter(
            JournalProfile.tfidf_vector.isnot(None),
            JournalProfile.bert_vector.isnot(None)
        ).all()
        
        if not profiles:
            return {'error': 'No profiles with vectors found'}
        
        tfidf_stats = {
            'dimensions': [],
            'non_zero_features': [],
            'norms': [],
            'mean_values': []
        }
        
        bert_stats = {
            'dimensions': [],
            'norms': [],
            'mean_values': [],
            'std_values': []
        }
        
        for profile in profiles[:100]:  # Sample first 100 to avoid performance issues
            try:
                # TF-IDF vector analysis
                tfidf_vec = np.array(json.loads(profile.tfidf_vector))
                tfidf_stats['dimensions'].append(len(tfidf_vec))
                tfidf_stats['non_zero_features'].append(np.count_nonzero(tfidf_vec))
                tfidf_stats['norms'].append(float(np.linalg.norm(tfidf_vec)))
                tfidf_stats['mean_values'].append(float(np.mean(tfidf_vec)))
                
                # BERT vector analysis
                bert_vec = np.array(json.loads(profile.bert_vector))
                bert_stats['dimensions'].append(len(bert_vec))
                bert_stats['norms'].append(float(np.linalg.norm(bert_vec)))
                bert_stats['mean_values'].append(float(np.mean(bert_vec)))
                bert_stats['std_values'].append(float(np.std(bert_vec)))
            except (json.JSONDecodeError, ValueError) as e:
                continue
        
        # Calculate aggregated statistics
        metrics = {
            'tfidf_vectors': {
                'avg_dimensions': float(np.mean(tfidf_stats['dimensions'])) if tfidf_stats['dimensions'] else 0,
                'avg_non_zero_features': float(np.mean(tfidf_stats['non_zero_features'])) if tfidf_stats['non_zero_features'] else 0,
                'avg_norm': float(np.mean(tfidf_stats['norms'])) if tfidf_stats['norms'] else 0,
                'avg_mean_value': float(np.mean(tfidf_stats['mean_values'])) if tfidf_stats['mean_values'] else 0,
                'sparsity': 1 - (float(np.mean(tfidf_stats['non_zero_features'])) / float(np.mean(tfidf_stats['dimensions']))) if tfidf_stats['dimensions'] else 0
            },
            'bert_vectors': {
                'avg_dimensions': float(np.mean(bert_stats['dimensions'])) if bert_stats['dimensions'] else 0,
                'avg_norm': float(np.mean(bert_stats['norms'])) if bert_stats['norms'] else 0,
                'avg_mean_value': float(np.mean(bert_stats['mean_values'])) if bert_stats['mean_values'] else 0,
                'avg_std_value': float(np.mean(bert_stats['std_values'])) if bert_stats['std_values'] else 0
            },
            'samples_analyzed': len(tfidf_stats['dimensions'])
        }
        
        # Detect anomalies (vectors with unexpected dimensions or zero norms)
        anomalies = {
            'zero_norm_tfidf': len([n for n in tfidf_stats['norms'] if n == 0]),
            'zero_norm_bert': len([n for n in bert_stats['norms'] if n == 0]),
            'unexpected_tfidf_dims': len([d for d in tfidf_stats['dimensions'] if d != tfidf_stats['dimensions'][0]]) if tfidf_stats['dimensions'] else 0,
            'unexpected_bert_dims': len([d for d in bert_stats['dimensions'] if d != bert_stats['dimensions'][0]]) if bert_stats['dimensions'] else 0
        }
        
        metrics['anomalies'] = anomalies
        
        return metrics
    
    def get_data_coverage_analysis(self) -> Dict:
        """
        Analyze completeness of journal data
        
        Returns:
            - Field completeness percentages
            - Missing data identification
            - Data quality scores
        """
        total_journals = self.db.query(Journal).count()
        
        if total_journals == 0:
            return {'error': 'No journals in database'}
        
        # Count completeness for each field
        field_counts = {
            'name': total_journals,  # Required field
            'display_name': self.db.query(Journal).filter(Journal.display_name.isnot(None)).count(),
            'issn': self.db.query(Journal).filter(Journal.issn.isnot(None)).count(),
            'eissn': self.db.query(Journal).filter(Journal.eissn.isnot(None)).count(),
            'publisher': self.db.query(Journal).filter(Journal.publisher.isnot(None)).count(),
            'impact_factor': self.db.query(Journal).filter(Journal.impact_factor.isnot(None)).count(),
            'subjects': self.db.query(Journal).filter(Journal.subjects.isnot(None)).count(),
            'profile': self.db.query(Journal).join(JournalProfile).count()
        }
        
        # Calculate completeness percentages
        completeness = {}
        for field, count in field_counts.items():
            completeness[field] = {
                'count': count,
                'percentage': round(count / total_journals * 100, 2)
            }
        
        # Calculate overall data quality score (0-100)
        # Weight different fields by importance
        weights = {
            'name': 1.0,
            'display_name': 0.5,
            'issn': 0.8,
            'eissn': 0.5,
            'publisher': 0.7,
            'impact_factor': 0.9,
            'subjects': 0.9,
            'profile': 1.0
        }
        
        weighted_score = sum(
            (count / total_journals) * weights[field] 
            for field, count in field_counts.items()
        )
        total_weight = sum(weights.values())
        quality_score = (weighted_score / total_weight) * 100
        
        return {
            'total_journals': total_journals,
            'field_completeness': completeness,
            'overall_quality_score': round(quality_score, 2),
            'missing_data_impact': {
                'journals_without_vectors': total_journals - field_counts['profile'],
                'journals_without_impact_factor': total_journals - field_counts['impact_factor'],
                'journals_without_subjects': total_journals - field_counts['subjects']
            }
        }
    
    def get_system_health_indicators(self) -> Dict:
        """
        Overall system health metrics
        
        Returns:
            - Health score (0-100)
            - Critical issues
            - Warnings
        """
        health_checks = {
            'database_accessible': True,
            'journals_present': False,
            'vectors_present': False,
            'recent_activity': False,
            'data_quality_acceptable': False
        }
        
        issues = []
        warnings = []
        
        # Check 1: Journals present
        journal_count = self.db.query(Journal).count()
        health_checks['journals_present'] = journal_count > 0
        if journal_count == 0:
            issues.append('No journals in database')
        elif journal_count < 100:
            warnings.append(f'Low journal count: {journal_count}')
        
        # Check 2: Vectors present
        vector_count = self.db.query(JournalProfile).filter(
            JournalProfile.tfidf_vector.isnot(None),
            JournalProfile.bert_vector.isnot(None)
        ).count()
        health_checks['vectors_present'] = vector_count > 0
        if vector_count == 0:
            issues.append('No ML vectors generated')
        
        vector_coverage = (vector_count / journal_count * 100) if journal_count > 0 else 0
        if vector_coverage < 80:
            warnings.append(f'Low vector coverage: {vector_coverage:.1f}%')
        
        # Check 3: Recent activity
        recent_query_count = self.db.query(QueryRun).filter(
            QueryRun.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        health_checks['recent_activity'] = recent_query_count > 0
        if recent_query_count == 0:
            warnings.append('No queries in last 24 hours')
        
        # Check 4: Data quality
        coverage_data = self.get_data_coverage_analysis()
        quality_score = coverage_data.get('overall_quality_score', 0)
        health_checks['data_quality_acceptable'] = quality_score > 60
        if quality_score < 50:
            issues.append(f'Low data quality score: {quality_score:.1f}')
        elif quality_score < 70:
            warnings.append(f'Moderate data quality: {quality_score:.1f}')
        
        # Calculate overall health score
        passed_checks = sum(1 for v in health_checks.values() if v)
        health_score = (passed_checks / len(health_checks)) * 100
        
        # Determine status
        if issues:
            status = 'CRITICAL'
        elif warnings:
            status = 'WARNING'
        else:
            status = 'HEALTHY'
        
        return {
            'status': status,
            'health_score': round(health_score, 2),
            'checks': health_checks,
            'issues': issues,
            'warnings': warnings,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_publisher_statistics(self) -> Dict:
        """
        Analyze publisher distribution
        """
        journals = self.db.query(Journal).filter(Journal.publisher.isnot(None)).all()
        
        if not journals:
            return {'error': 'No publisher data available'}
        
        publisher_counts = {}
        for journal in journals:
            publisher = journal.publisher
            publisher_counts[publisher] = publisher_counts.get(publisher, 0) + 1
        
        # Sort by count
        sorted_publishers = sorted(publisher_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_publishers': len(publisher_counts),
            'top_10_publishers': dict(sorted_publishers[:10]),
            'publishers_with_single_journal': len([p for p, count in publisher_counts.items() if count == 1]),
            'avg_journals_per_publisher': round(len(journals) / len(publisher_counts), 2)
        }
    
    def export_metrics(self, filepath: str):
        """Export system metrics to JSON file"""
        metrics = {
            'database_statistics': self.get_database_statistics(),
            'vector_quality': self.get_vector_quality_metrics(),
            'data_coverage': self.get_data_coverage_analysis(),
            'system_health': self.get_system_health_indicators(),
            'publisher_statistics': self.get_publisher_statistics(),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db'):
            self.db.close()
