"""
Performance Metrics Module

Tracks and analyzes system performance including:
- Response time statistics
- Throughput metrics
- Latency distribution
- Query processing time breakdown
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.models.base import SessionLocal
from app.models.entities import QueryRun, Recommendation
from sqlalchemy import func


class PerformanceMetrics:
    """Track and analyze performance metrics"""
    
    def __init__(self):
        self.db = SessionLocal()
        self._request_times = []
        self._component_times = defaultdict(list)
    
    def record_request_time(self, duration_ms: float, endpoint: str = "recommend"):
        """Record a single request processing time"""
        self._request_times.append({
            'timestamp': datetime.utcnow(),
            'duration_ms': duration_ms,
            'endpoint': endpoint
        })
    
    def record_component_time(self, component: str, duration_ms: float):
        """Record time taken by individual components (BERT, TF-IDF, etc.)"""
        self._component_times[component].append(duration_ms)
    
    def get_response_time_stats(self, hours: int = 24) -> Dict:
        """
        Get response time statistics for the specified time window
        
        Returns:
            - mean, median, p95, p99 response times
            - min/max response times
            - standard deviation
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_times = [
            r['duration_ms'] for r in self._request_times 
            if r['timestamp'] > cutoff
        ]
        
        if not recent_times:
            return self._empty_stats()
        
        return {
            'mean_ms': float(np.mean(recent_times)),
            'median_ms': float(np.median(recent_times)),
            'std_ms': float(np.std(recent_times)),
            'min_ms': float(np.min(recent_times)),
            'max_ms': float(np.max(recent_times)),
            'p95_ms': float(np.percentile(recent_times, 95)),
            'p99_ms': float(np.percentile(recent_times, 99)),
            'total_requests': len(recent_times),
            'time_window_hours': hours
        }
    
    def get_throughput_metrics(self, hours: int = 24) -> Dict:
        """
        Calculate throughput metrics
        
        Returns:
            - Requests per minute/hour
            - Peak throughput periods
            - Average query rate
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Get queries from database
        query_count = self.db.query(QueryRun).filter(
            QueryRun.timestamp >= cutoff
        ).count()
        
        # Calculate rates
        total_minutes = hours * 60
        total_seconds = hours * 3600
        
        return {
            'total_queries': query_count,
            'time_window_hours': hours,
            'queries_per_minute': round(query_count / total_minutes, 2),
            'queries_per_hour': round(query_count / hours, 2),
            'queries_per_second': round(query_count / total_seconds, 4),
            'avg_throughput': f"{round(query_count / hours, 2)} queries/hour"
        }
    
    def get_latency_distribution(self) -> Dict:
        """
        Analyze latency distribution across different percentiles
        
        Returns:
            - Distribution histogram
            - Percentile breakdown (p50, p75, p90, p95, p99)
        """
        if not self._request_times:
            return {'error': 'No timing data available'}
        
        times = [r['duration_ms'] for r in self._request_times]
        
        # Create histogram bins
        bins = [0, 50, 100, 200, 500, 1000, 2000, 5000]
        hist, _ = np.histogram(times, bins=bins)
        
        distribution = {}
        for i in range(len(bins)-1):
            label = f"{bins[i]}-{bins[i+1]}ms"
            distribution[label] = int(hist[i])
        distribution[f">{bins[-1]}ms"] = int(np.sum(np.array(times) > bins[-1]))
        
        return {
            'distribution': distribution,
            'percentiles': {
                'p25': float(np.percentile(times, 25)),
                'p50': float(np.percentile(times, 50)),
                'p75': float(np.percentile(times, 75)),
                'p90': float(np.percentile(times, 90)),
                'p95': float(np.percentile(times, 95)),
                'p99': float(np.percentile(times, 99)),
            },
            'total_samples': len(times)
        }
    
    def get_component_breakdown(self) -> Dict:
        """
        Analyze time spent in different system components
        
        Returns:
            - Time breakdown by component (BERT, TF-IDF, DB, etc.)
            - Percentage contribution to total time
        """
        breakdown = {}
        total_time = 0
        
        for component, times in self._component_times.items():
            if times:
                component_total = sum(times)
                total_time += component_total
                breakdown[component] = {
                    'total_ms': round(component_total, 2),
                    'avg_ms': round(np.mean(times), 2),
                    'count': len(times)
                }
        
        # Add percentages
        if total_time > 0:
            for component in breakdown:
                breakdown[component]['percentage'] = round(
                    (breakdown[component]['total_ms'] / total_time) * 100, 2
                )
        
        return {
            'component_breakdown': breakdown,
            'total_time_ms': round(total_time, 2)
        }
    
    def get_slow_queries(self, threshold_ms: float = 1000, limit: int = 10) -> List[Dict]:
        """
        Identify slow queries above threshold
        
        Args:
            threshold_ms: Minimum duration to be considered slow
            limit: Maximum number of queries to return
        """
        slow_queries = [
            {
                'timestamp': r['timestamp'].isoformat(),
                'duration_ms': r['duration_ms'],
                'endpoint': r['endpoint']
            }
            for r in self._request_times 
            if r['duration_ms'] > threshold_ms
        ]
        
        # Sort by duration (slowest first)
        slow_queries.sort(key=lambda x: x['duration_ms'], reverse=True)
        
        return slow_queries[:limit]
    
    def get_query_processing_breakdown(self) -> Dict:
        """
        Detailed breakdown of query processing stages
        
        Returns:
            - Average time per stage (encoding, similarity calculation, ranking, DB)
        """
        stages = {
            'text_encoding': self._component_times.get('text_encoding', []),
            'vector_similarity': self._component_times.get('vector_similarity', []),
            'ranking': self._component_times.get('ranking', []),
            'database_operations': self._component_times.get('database', []),
            'response_formatting': self._component_times.get('formatting', [])
        }
        
        breakdown = {}
        for stage, times in stages.items():
            if times:
                breakdown[stage] = {
                    'avg_ms': round(np.mean(times), 2),
                    'median_ms': round(np.median(times), 2),
                    'max_ms': round(np.max(times), 2),
                    'count': len(times)
                }
            else:
                breakdown[stage] = {
                    'avg_ms': 0,
                    'median_ms': 0,
                    'max_ms': 0,
                    'count': 0
                }
        
        return breakdown
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics structure"""
        return {
            'mean_ms': 0,
            'median_ms': 0,
            'std_ms': 0,
            'min_ms': 0,
            'max_ms': 0,
            'p95_ms': 0,
            'p99_ms': 0,
            'total_requests': 0
        }
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to JSON file"""
        metrics = {
            'response_time_stats': self.get_response_time_stats(),
            'throughput_metrics': self.get_throughput_metrics(),
            'latency_distribution': self.get_latency_distribution(),
            'component_breakdown': self.get_component_breakdown(),
            'slow_queries': self.get_slow_queries(),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db'):
            self.db.close()
