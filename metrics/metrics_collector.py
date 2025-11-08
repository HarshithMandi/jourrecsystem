"""
Metrics Collector

Central module for collecting and aggregating all metrics.
Provides unified interface for metrics collection and export.
"""

import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from .performance_metrics import PerformanceMetrics
from .accuracy_metrics import AccuracyMetrics
from .system_metrics import SystemMetrics
from .user_metrics import UserMetrics


class MetricsCollector:
    """
    Central metrics collection and aggregation system
    
    Usage:
        collector = MetricsCollector()
        
        # Collect all metrics
        all_metrics = collector.collect_all_metrics()
        
        # Export to file
        collector.export_all_metrics('metrics_report.json')
        
        # Get specific category
        perf_metrics = collector.get_performance_metrics()
    """
    
    def __init__(self):
        self.performance = PerformanceMetrics()
        self.accuracy = AccuracyMetrics()
        self.system = SystemMetrics()
        self.user = UserMetrics()
    
    def collect_all_metrics(self, hours: int = 24) -> Dict:
        """
        Collect all metrics from all categories
        
        Args:
            hours: Time window for time-based metrics
        
        Returns:
            Comprehensive metrics dictionary
        """
        return {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'time_window_hours': hours,
                'version': '1.0'
            },
            'performance': self.get_performance_metrics(hours),
            'accuracy': self.get_accuracy_metrics(hours),
            'system': self.get_system_metrics(),
            'user': self.get_user_metrics(hours)
        }
    
    def get_performance_metrics(self, hours: int = 24) -> Dict:
        """Get all performance metrics"""
        return {
            'response_time': self.performance.get_response_time_stats(hours),
            'throughput': self.performance.get_throughput_metrics(hours),
            'latency_distribution': self.performance.get_latency_distribution(),
            'component_breakdown': self.performance.get_component_breakdown(),
            'slow_queries': self.performance.get_slow_queries(threshold_ms=1000, limit=10)
        }
    
    def get_accuracy_metrics(self, hours: int = 24) -> Dict:
        """Get all accuracy metrics"""
        return {
            'similarity_distribution': self.accuracy.get_similarity_score_distribution(hours),
            'ranking_quality': self.accuracy.get_ranking_quality_metrics(hours),
            'recommendation_diversity': self.accuracy.get_recommendation_diversity(hours)
        }
    
    def get_system_metrics(self) -> Dict:
        """Get all system metrics"""
        return {
            'database_stats': self.system.get_database_statistics(),
            'vector_quality': self.system.get_vector_quality_metrics(),
            'data_coverage': self.system.get_data_coverage_analysis(),
            'system_health': self.system.get_system_health_indicators(),
            'publisher_stats': self.system.get_publisher_statistics()
        }
    
    def get_user_metrics(self, hours: int = 168) -> Dict:
        """Get all user behavior metrics"""
        return {
            'query_patterns': self.user.get_query_patterns(hours),
            'popular_journals': self.user.get_popular_journals(limit=20, hours=hours),
            'topic_trends': self.user.get_topic_trends(hours),
            'interaction_patterns': self.user.get_user_interaction_patterns(),
            'open_access_preference': self.user.get_open_access_preference(hours)
        }
    
    def get_summary_dashboard(self) -> Dict:
        """
        Get high-level summary for dashboard display
        
        Returns:
            Key metrics in concise format
        """
        perf = self.performance.get_response_time_stats(24)
        sys_health = self.system.get_system_health_indicators()
        db_stats = self.system.get_database_statistics()
        query_patterns = self.user.get_query_patterns(24)
        
        return {
            'system_status': sys_health['status'],
            'health_score': sys_health['health_score'],
            'total_journals': db_stats['journals']['total'],
            'vector_coverage': db_stats.get('coverage', {}).get('vector_coverage', 0),
            'queries_24h': query_patterns.get('total_queries', 0),
            'avg_response_time_ms': perf['mean_ms'],
            'p95_response_time_ms': perf['p95_ms'],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def export_all_metrics(self, filepath: str, hours: int = 24):
        """
        Export all metrics to JSON file
        
        Args:
            filepath: Path to output JSON file
            hours: Time window for metrics
        """
        metrics = self.collect_all_metrics(hours)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Metrics exported to: {filepath}")
    
    def export_summary_report(self, filepath: str):
        """
        Export concise summary report
        
        Args:
            filepath: Path to output text file
        """
        summary = self.get_summary_dashboard()
        perf = self.get_performance_metrics(24)
        accuracy = self.get_accuracy_metrics(24)
        
        report_lines = [
            "=" * 60,
            "JOURNAL RECOMMENDATION SYSTEM - METRICS SUMMARY",
            "=" * 60,
            f"Generated: {summary['timestamp']}",
            "",
            "SYSTEM HEALTH",
            "-" * 60,
            f"Status: {summary['system_status']}",
            f"Health Score: {summary['health_score']}/100",
            f"Total Journals: {summary['total_journals']}",
            f"Vector Coverage: {summary['vector_coverage']}%",
            "",
            "PERFORMANCE (Last 24 Hours)",
            "-" * 60,
            f"Total Queries: {summary['queries_24h']}",
            f"Avg Response Time: {summary['avg_response_time_ms']:.2f} ms",
            f"P95 Response Time: {summary['p95_response_time_ms']:.2f} ms",
            f"Throughput: {perf['throughput']['queries_per_hour']:.2f} queries/hour",
            "",
            "ACCURACY",
            "-" * 60,
        ]
        
        if 'statistics' in accuracy['similarity_distribution']:
            stats = accuracy['similarity_distribution']['statistics']
            report_lines.extend([
                f"Avg Similarity Score: {stats['mean']:.3f}",
                f"Median Similarity Score: {stats['median']:.3f}",
                f"High Quality Recs (>0.7): {accuracy['similarity_distribution']['high_quality_recommendations']}",
            ])
        
        report_lines.append("=" * 60)
        
        # Write to file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report exported to: {filepath}")
    
    def record_request(self, duration_ms: float, endpoint: str = "recommend"):
        """
        Record a request for performance tracking
        
        Args:
            duration_ms: Request duration in milliseconds
            endpoint: API endpoint name
        """
        self.performance.record_request_time(duration_ms, endpoint)
    
    def record_component_time(self, component: str, duration_ms: float):
        """
        Record component execution time
        
        Args:
            component: Component name (bert, tfidf, database, etc.)
            duration_ms: Duration in milliseconds
        """
        self.performance.record_component_time(component, duration_ms)
