"""
Accuracy Visualization Module

Creates visualizations for accuracy metrics:
- Similarity score distributions
- Ranking quality analysis
- Component contribution charts
- Diversity metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
from pathlib import Path


class AccuracyVisualizer:
    """Generate accuracy metric visualizations"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        self.figure_size = (12, 6)
    
    def plot_similarity_distribution(self, metrics: Dict, output_path: str):
        """
        Plot similarity score distribution
        
        Args:
            metrics: Accuracy metrics dictionary
            output_path: Path to save figure
        """
        sim_dist = metrics.get('similarity_distribution', {})
        distribution = sim_dist.get('distribution', {})
        statistics = sim_dist.get('statistics', {})
        
        if not distribution:
            print("No similarity distribution data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        bins = list(distribution.keys())
        counts = list(distribution.values())
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bins)))
        ax1.bar(range(len(bins)), counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Similarity Score Range')
        ax1.set_ylabel('Number of Recommendations')
        ax1.set_title('Similarity Score Distribution')
        ax1.set_xticks(range(len(bins)))
        ax1.set_xticklabels(bins, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Statistics box
        if statistics:
            stat_text = f"""
            Mean: {statistics.get('mean', 0):.3f}
            Median: {statistics.get('median', 0):.3f}
            Std Dev: {statistics.get('std', 0):.3f}
            Min: {statistics.get('min', 0):.3f}
            Max: {statistics.get('max', 0):.3f}
            Q25: {statistics.get('q25', 0):.3f}
            Q75: {statistics.get('q75', 0):.3f}
            """
            
            ax2.axis('off')
            ax2.text(0.1, 0.5, stat_text, fontsize=12, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    verticalalignment='center')
            ax2.set_title('Statistics Summary')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Similarity distribution saved to: {output_path}")
    
    def plot_quality_breakdown(self, metrics: Dict, output_path: str):
        """
        Plot recommendation quality breakdown (high/medium/low)
        
        Args:
            metrics: Accuracy metrics dictionary
            output_path: Path to save figure
        """
        sim_dist = metrics.get('similarity_distribution', {})
        
        high = sim_dist.get('high_quality_recommendations', 0)
        medium = sim_dist.get('medium_quality_recommendations', 0)
        low = sim_dist.get('low_quality_recommendations', 0)
        
        if high + medium + low == 0:
            print("No quality data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        sizes = [high, medium, low]
        labels = ['High Quality\n(>0.7)', 'Medium Quality\n(0.4-0.7)', 'Low Quality\n(<0.4)']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        explode = (0.05, 0, 0)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        ax1.set_title('Recommendation Quality Distribution')
        
        # Bar chart
        ax2.bar(labels, sizes, color=colors, alpha=0.7)
        ax2.set_ylabel('Number of Recommendations')
        ax2.set_title('Quality Breakdown')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(sizes):
            ax2.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Quality breakdown saved to: {output_path}")
    
    def plot_ranking_quality(self, metrics: Dict, output_path: str):
        """
        Plot ranking quality metrics
        
        Args:
            metrics: Accuracy metrics dictionary
            output_path: Path to save figure
        """
        ranking = metrics.get('ranking_quality', {})
        
        if not ranking or 'error' in ranking:
            print("No ranking quality data available")
            return
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create comparison of top-K scores
        metric_names = ['Top-1 Avg', 'Top-3 Avg', 'Top-5 Avg']
        values = [
            ranking.get('top1_avg_score', 0),
            ranking.get('top3_avg_score', 0),
            ranking.get('top5_avg_score', 0)
        ]
        
        colors = ['#e74c3c', '#f39c12', '#3498db']
        bars = ax.bar(metric_names, values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Average Similarity Score')
        ax.set_title('Top-K Ranking Quality')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add consistency metric as text
        consistency = ranking.get('ranking_consistency', 0)
        ax.text(0.98, 0.98, f'Ranking Consistency: {consistency:.3f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Ranking quality saved to: {output_path}")
    
    def plot_diversity_metrics(self, metrics: Dict, output_path: str):
        """
        Plot recommendation diversity metrics
        
        Args:
            metrics: Accuracy metrics dictionary
            output_path: Path to save figure
        """
        diversity = metrics.get('recommendation_diversity', {})
        
        if not diversity or 'error' in diversity:
            print("No diversity data available")
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Open access ratio
        ax1 = fig.add_subplot(gs[0, 0])
        oa_ratio = diversity.get('open_access_ratio', 0)
        trad_ratio = 1 - oa_ratio
        
        ax1.pie([oa_ratio, trad_ratio],
               labels=['Open Access', 'Traditional'],
               colors=['#2ecc71', '#95a5a6'],
               autopct='%1.1f%%',
               startangle=90)
        ax1.set_title('Open Access vs Traditional')
        
        # 2. Publisher diversity
        ax2 = fig.add_subplot(gs[0, 1])
        top_publishers = diversity.get('top_publishers', {})
        if top_publishers:
            pubs = list(top_publishers.keys())[:5]
            counts = list(top_publishers.values())[:5]
            
            ax2.barh(pubs, counts, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Recommendation Count')
            ax2.set_title('Top 5 Publishers')
            ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Subject diversity
        ax3 = fig.add_subplot(gs[1, :])
        top_subjects = diversity.get('top_subjects', {})
        if top_subjects:
            subjects = list(top_subjects.keys())[:10]
            counts = list(top_subjects.values())[:10]
            
            ax3.bar(range(len(subjects)), counts, color='coral', alpha=0.7)
            ax3.set_xlabel('Subject Area')
            ax3.set_ylabel('Recommendation Count')
            ax3.set_title('Top 10 Subject Areas')
            ax3.set_xticks(range(len(subjects)))
            ax3.set_xticklabels(subjects, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary stats
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        summary_text = f"""
        Diversity Metrics Summary:
        
        Unique Journals Recommended: {diversity.get('unique_journals_recommended', 0)}
        Unique Publishers: {diversity.get('unique_publishers', 0)}
        Subject Diversity: {diversity.get('subject_diversity', 0)} unique subjects
        Open Access Count: {diversity.get('open_access_count', 0)}
        Traditional Count: {diversity.get('traditional_count', 0)}
        """
        
        ax4.text(0.5, 0.5, summary_text, fontsize=11, family='monospace',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Diversity metrics saved to: {output_path}")
    
    def generate_all_accuracy_plots(self, metrics: Dict, output_dir: str):
        """
        Generate all accuracy visualizations
        
        Args:
            metrics: Accuracy metrics dictionary
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating accuracy visualizations...")
        
        self.plot_similarity_distribution(metrics, str(output_path / "similarity_distribution.png"))
        self.plot_quality_breakdown(metrics, str(output_path / "quality_breakdown.png"))
        self.plot_ranking_quality(metrics, str(output_path / "ranking_quality.png"))
        self.plot_diversity_metrics(metrics, str(output_path / "diversity_metrics.png"))
        
        print(f"All accuracy visualizations saved to: {output_dir}")
