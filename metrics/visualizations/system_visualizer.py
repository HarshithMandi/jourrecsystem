"""
System Visualization Module

Creates visualizations for system metrics:
- Database statistics
- Vector quality analysis
- Data coverage heatmaps
- System health dashboards
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
from pathlib import Path


class SystemVisualizer:
    """Generate system metric visualizations"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        self.figure_size = (12, 6)
    
    def plot_database_stats(self, metrics: Dict, output_path: str):
        """
        Plot database statistics overview
        
        Args:
            metrics: System metrics dictionary
            output_path: Path to save figure
        """
        db_stats = metrics.get('database_stats', {})
        
        if not db_stats:
            print("No database stats available")
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Journal counts
        ax1 = fig.add_subplot(gs[0, 0])
        journals = db_stats.get('journals', {})
        
        categories = ['Total', 'With Profiles', 'Open Access', 'With Impact', 'With ISSN']
        values = [
            journals.get('total', 0),
            journals.get('with_profiles', 0),
            journals.get('open_access', 0),
            journals.get('with_impact_factor', 0),
            journals.get('with_issn', 0)
        ]
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        ax1.bar(range(len(categories)), values, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.set_ylabel('Count')
        ax1.set_title('Journal Database Statistics')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Profile coverage
        ax2 = fig.add_subplot(gs[0, 1])
        profiles = db_stats.get('profiles', {})
        
        profile_categories = ['Total', 'TF-IDF', 'BERT', 'Both']
        profile_values = [
            profiles.get('total', 0),
            profiles.get('with_tfidf_vectors', 0),
            profiles.get('with_bert_vectors', 0),
            profiles.get('with_both_vectors', 0)
        ]
        
        ax2.bar(range(len(profile_categories)), profile_values, color='teal', alpha=0.7)
        ax2.set_xticks(range(len(profile_categories)))
        ax2.set_xticklabels(profile_categories)
        ax2.set_ylabel('Count')
        ax2.set_title('Vector Profile Coverage')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Coverage percentages
        ax3 = fig.add_subplot(gs[1, :])
        coverage = db_stats.get('coverage', {})
        
        if coverage:
            cov_categories = list(coverage.keys())
            cov_values = list(coverage.values())
            
            bars = ax3.barh(cov_categories, cov_values, color='coral', alpha=0.7)
            ax3.set_xlabel('Coverage (%)')
            ax3.set_title('Data Coverage Percentages')
            ax3.set_xlim([0, 100])
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add percentage labels
            for bar, value in zip(bars, cov_values):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{value:.1f}%',
                        ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 4. Queries and recommendations
        ax4 = fig.add_subplot(gs[2, 0])
        queries = db_stats.get('queries', {})
        recommendations = db_stats.get('recommendations', {})
        
        activity_data = [
            ('Total Queries', queries.get('total', 0)),
            ('Queries with Recs', queries.get('with_recommendations', 0)),
            ('Total Recommendations', recommendations.get('total', 0))
        ]
        
        labels = [x[0] for x in activity_data]
        values = [x[1] for x in activity_data]
        
        ax4.bar(range(len(labels)), values, color=['#3498db', '#2ecc71', '#f39c12'], alpha=0.7)
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_ylabel('Count')
        ax4.set_title('Query & Recommendation Activity')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Works statistics
        ax5 = fig.add_subplot(gs[2, 1])
        works = db_stats.get('works', {})
        
        work_data = [
            ('Total Works', works.get('total', 0)),
            ('With Abstracts', works.get('with_abstracts', 0))
        ]
        
        labels = [x[0] for x in work_data]
        values = [x[1] for x in work_data]
        
        ax5.bar(range(len(labels)), values, color=['#9b59b6', '#e67e22'], alpha=0.7)
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels)
        ax5.set_ylabel('Count')
        ax5.set_title('Research Works Statistics')
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Database stats saved to: {output_path}")
    
    def plot_vector_quality(self, metrics: Dict, output_path: str):
        """
        Plot vector quality metrics
        
        Args:
            metrics: System metrics dictionary
            output_path: Path to save figure
        """
        vector_quality = metrics.get('vector_quality', {})
        
        if not vector_quality or 'error' in vector_quality:
            print("No vector quality data available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. TF-IDF vector metrics
        tfidf = vector_quality.get('tfidf_vectors', {})
        if tfidf:
            tfidf_metrics = ['Avg Dimensions', 'Avg Non-Zero', 'Avg Norm', 'Sparsity']
            tfidf_values = [
                tfidf.get('avg_dimensions', 0),
                tfidf.get('avg_non_zero_features', 0),
                tfidf.get('avg_norm', 0),
                tfidf.get('sparsity', 0) * 100  # Convert to percentage
            ]
            
            ax1.bar(range(len(tfidf_metrics)), tfidf_values, color='steelblue', alpha=0.7)
            ax1.set_xticks(range(len(tfidf_metrics)))
            ax1.set_xticklabels(tfidf_metrics, rotation=45, ha='right')
            ax1.set_title('TF-IDF Vector Quality Metrics')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. BERT vector metrics
        bert = vector_quality.get('bert_vectors', {})
        if bert:
            bert_metrics = ['Avg Dimensions', 'Avg Norm', 'Avg Mean', 'Avg Std']
            bert_values = [
                bert.get('avg_dimensions', 0),
                bert.get('avg_norm', 0),
                bert.get('avg_mean_value', 0),
                bert.get('avg_std_value', 0)
            ]
            
            ax2.bar(range(len(bert_metrics)), bert_values, color='coral', alpha=0.7)
            ax2.set_xticks(range(len(bert_metrics)))
            ax2.set_xticklabels(bert_metrics, rotation=45, ha='right')
            ax2.set_title('BERT Vector Quality Metrics')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Anomalies
        anomalies = vector_quality.get('anomalies', {})
        if anomalies:
            anomaly_types = list(anomalies.keys())
            anomaly_counts = list(anomalies.values())
            
            colors = ['#e74c3c' if count > 0 else '#2ecc71' for count in anomaly_counts]
            ax3.barh(anomaly_types, anomaly_counts, color=colors, alpha=0.7)
            ax3.set_xlabel('Count')
            ax3.set_title('Vector Anomalies Detection')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Summary info
        ax4.axis('off')
        summary_text = f"""
        Vector Quality Summary:
        
        Samples Analyzed: {vector_quality.get('samples_analyzed', 0)}
        
        TF-IDF Vectors:
          - Dimensions: {tfidf.get('avg_dimensions', 0):.0f}
          - Sparsity: {tfidf.get('sparsity', 0)*100:.1f}%
        
        BERT Vectors:
          - Dimensions: {bert.get('avg_dimensions', 0):.0f}
          - Avg Norm: {bert.get('avg_norm', 0):.2f}
        """
        
        ax4.text(0.5, 0.5, summary_text, fontsize=10, family='monospace',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Vector quality saved to: {output_path}")
    
    def plot_system_health(self, metrics: Dict, output_path: str):
        """
        Plot system health dashboard
        
        Args:
            metrics: System metrics dictionary
            output_path: Path to save figure
        """
        health = metrics.get('system_health', {})
        
        if not health:
            print("No system health data available")
            return
        
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Health score gauge
        ax1 = fig.add_subplot(gs[0, :])
        health_score = health.get('health_score', 0)
        status = health.get('status', 'UNKNOWN')
        
        # Create gauge visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color based on health score
        if health_score >= 80:
            color = '#2ecc71'
        elif health_score >= 60:
            color = '#f39c12'
        else:
            color = '#e74c3c'
        
        ax1 = plt.subplot(gs[0, :], projection='polar')
        ax1.plot(theta, r, 'k-', linewidth=2)
        ax1.fill_between(theta, 0, r, alpha=0.1, color='gray')
        
        # Fill up to health score
        theta_fill = np.linspace(0, np.pi * (health_score / 100), 100)
        ax1.fill_between(theta_fill, 0, r[:len(theta_fill)], alpha=0.7, color=color)
        
        ax1.set_ylim([0, 1])
        ax1.set_yticks([])
        ax1.set_xticks([])
        ax1.text(0, 0, f"{health_score:.0f}%\n{status}", 
                ha='center', va='center', fontsize=20, fontweight='bold')
        ax1.set_title('System Health Score', pad=20)
        
        # 2. Health checks
        ax2 = fig.add_subplot(gs[1, 0])
        checks = health.get('checks', {})
        
        if checks:
            check_names = list(checks.keys())
            check_status = [1 if v else 0 for v in checks.values()]
            colors = ['#2ecc71' if v else '#e74c3c' for v in check_status]
            
            ax2.barh(check_names, check_status, color=colors, alpha=0.7)
            ax2.set_xlabel('Status (1=Pass, 0=Fail)')
            ax2.set_title('Health Checks')
            ax2.set_xlim([0, 1.2])
            ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Issues and warnings
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        issues = health.get('issues', [])
        warnings = health.get('warnings', [])
        
        alert_text = "Issues:\n"
        if issues:
            for issue in issues:
                alert_text += f"  ❌ {issue}\n"
        else:
            alert_text += "  ✓ None\n"
        
        alert_text += "\nWarnings:\n"
        if warnings:
            for warning in warnings:
                alert_text += f"  ⚠️  {warning}\n"
        else:
            alert_text += "  ✓ None\n"
        
        ax3.text(0.1, 0.5, alert_text, fontsize=9, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax3.set_title('Issues & Warnings')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"System health saved to: {output_path}")
    
    def generate_all_system_plots(self, metrics: Dict, output_dir: str):
        """
        Generate all system visualizations
        
        Args:
            metrics: System metrics dictionary
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating system visualizations...")
        
        self.plot_database_stats(metrics, str(output_path / "database_stats.png"))
        self.plot_vector_quality(metrics, str(output_path / "vector_quality.png"))
        self.plot_system_health(metrics, str(output_path / "system_health.png"))
        
        print(f"All system visualizations saved to: {output_dir}")
