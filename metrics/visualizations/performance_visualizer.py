"""
Performance Visualization Module

Creates visualizations for performance metrics:
- Response time charts
- Latency distributions
- Throughput graphs
- Component breakdown
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class PerformanceVisualizer:
    """Generate performance metric visualizations"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        self.figure_size = (12, 6)
    
    def plot_response_time_distribution(self, metrics: Dict, output_path: str):
        """
        Plot response time distribution histogram
        
        Args:
            metrics: Performance metrics dictionary
            output_path: Path to save figure
        """
        latency_dist = metrics.get('latency_distribution', {})
        distribution = latency_dist.get('distribution', {})
        
        if not distribution:
            print("No latency distribution data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of latency distribution
        bins = list(distribution.keys())
        counts = list(distribution.values())
        
        ax1.bar(range(len(bins)), counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Response Time Range')
        ax1.set_ylabel('Number of Requests')
        ax1.set_title('Response Time Distribution')
        ax1.set_xticks(range(len(bins)))
        ax1.set_xticklabels(bins, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Box plot of percentiles
        percentiles = latency_dist.get('percentiles', {})
        if percentiles:
            values = list(percentiles.values())
            labels = list(percentiles.keys())
            
            ax2.boxplot([values], vert=False, widths=0.5)
            ax2.set_yticklabels(['Response Time'])
            ax2.set_xlabel('Time (ms)')
            ax2.set_title('Response Time Percentiles')
            ax2.grid(True, alpha=0.3)
            
            # Add percentile markers
            for i, (label, value) in enumerate(percentiles.items()):
                ax2.axvline(value, color='red', linestyle='--', alpha=0.3)
                ax2.text(value, 1.05, label, fontsize=8, ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Response time distribution saved to: {output_path}")
    
    def plot_throughput_timeline(self, metrics: Dict, output_path: str):
        """
        Plot throughput over time
        
        Args:
            metrics: Performance metrics dictionary
            output_path: Path to save figure
        """
        throughput = metrics.get('throughput', {})
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create visualization of throughput metrics
        metric_names = ['Queries/Second', 'Queries/Minute', 'Queries/Hour']
        values = [
            throughput.get('queries_per_second', 0),
            throughput.get('queries_per_minute', 0),
            throughput.get('queries_per_hour', 0)
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax.barh(metric_names, values, color=colors, alpha=0.7)
        ax.set_xlabel('Query Rate')
        ax.set_title(f"Throughput Metrics ({throughput.get('time_window_hours', 24)}h window)")
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v, i, f' {v:.2f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Throughput timeline saved to: {output_path}")
    
    def plot_component_breakdown(self, metrics: Dict, output_path: str):
        """
        Plot component execution time breakdown
        
        Args:
            metrics: Performance metrics dictionary
            output_path: Path to save figure
        """
        component_data = metrics.get('component_breakdown', {})
        breakdown = component_data.get('component_breakdown', {})
        
        if not breakdown:
            print("No component breakdown data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of time distribution
        components = list(breakdown.keys())
        times = [data['total_ms'] for data in breakdown.values()]
        
        colors = sns.color_palette("Set3", len(components))
        
        ax1.pie(times, labels=components, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Component Time Distribution')
        
        # Bar chart of average times
        avg_times = [data['avg_ms'] for data in breakdown.values()]
        
        ax2.barh(components, avg_times, color=colors, alpha=0.7)
        ax2.set_xlabel('Average Time (ms)')
        ax2.set_title('Average Execution Time by Component')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Component breakdown saved to: {output_path}")
    
    def plot_slow_queries(self, metrics: Dict, output_path: str):
        """
        Visualize slow queries
        
        Args:
            metrics: Performance metrics dictionary
            output_path: Path to save figure
        """
        slow_queries = metrics.get('slow_queries', [])
        
        if not slow_queries:
            print("No slow queries to visualize")
            return
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Extract data
        durations = [q['duration_ms'] for q in slow_queries]
        endpoints = [q.get('endpoint', 'unknown') for q in slow_queries]
        indices = range(len(durations))
        
        # Color by endpoint
        unique_endpoints = list(set(endpoints))
        colors_map = dict(zip(unique_endpoints, sns.color_palette("Set2", len(unique_endpoints))))
        colors = [colors_map[ep] for ep in endpoints]
        
        ax.barh(indices, durations, color=colors, alpha=0.7)
        ax.set_xlabel('Duration (ms)')
        ax.set_ylabel('Query Index')
        ax.set_title(f'Top {len(slow_queries)} Slowest Queries')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors_map[ep], label=ep) for ep in unique_endpoints]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Slow queries visualization saved to: {output_path}")
    
    def plot_response_time_stats(self, metrics: Dict, output_path: str):
        """
        Plot response time statistics summary
        
        Args:
            metrics: Performance metrics dictionary
            output_path: Path to save figure
        """
        response_time = metrics.get('response_time', {})
        
        if not response_time or response_time.get('total_requests', 0) == 0:
            print("No response time data available")
            return
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create bar chart of key metrics
        stat_names = ['Mean', 'Median', 'P95', 'P99', 'Max']
        values = [
            response_time.get('mean_ms', 0),
            response_time.get('median_ms', 0),
            response_time.get('p95_ms', 0),
            response_time.get('p99_ms', 0),
            response_time.get('max_ms', 0)
        ]
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        bars = ax.bar(stat_names, values, color=colors, alpha=0.7)
        
        ax.set_ylabel('Time (ms)')
        ax.set_title(f"Response Time Statistics ({response_time.get('total_requests', 0)} requests)")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}ms',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Response time stats saved to: {output_path}")
    
    def generate_all_performance_plots(self, metrics: Dict, output_dir: str):
        """
        Generate all performance visualizations
        
        Args:
            metrics: Performance metrics dictionary
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating performance visualizations...")
        
        self.plot_response_time_stats(metrics, str(output_path / "response_time_stats.png"))
        self.plot_response_time_distribution(metrics, str(output_path / "response_time_distribution.png"))
        self.plot_throughput_timeline(metrics, str(output_path / "throughput.png"))
        self.plot_component_breakdown(metrics, str(output_path / "component_breakdown.png"))
        self.plot_slow_queries(metrics, str(output_path / "slow_queries.png"))
        
        print(f"All performance visualizations saved to: {output_dir}")
