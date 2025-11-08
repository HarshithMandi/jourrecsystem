"""
Ground Truth Validation Visualizations

Creates comprehensive comparison charts for:
- Hybrid (TF-IDF + BERT)
- TF-IDF Only  
- BERT Only

Metrics visualized:
- Hit Rate @ K=10, K=20
- Mean Reciprocal Rank (MRR)
- Average Rank
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict


class GroundTruthVisualizer:
    """Create visualizations for ground truth validation results."""
    
    def __init__(self):
        sns.set_style("whitegrid")
        self.colors = {
            'hybrid': '#667eea',
            'tfidf': '#f093fb',
            'bert': '#4facfe'
        }
        self.model_labels = {
            'hybrid': 'Hybrid\n(TF-IDF + BERT)',
            'tfidf': 'TF-IDF\nOnly',
            'bert': 'BERT\nOnly'
        }
    
    def plot_hit_rate_comparison(self, summary: Dict, output_path: str = None):
        """
        Create grouped bar chart comparing hit rates @ K=10 and K=20.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        models = ['hybrid', 'tfidf', 'bert']
        k_values = ['top_10', 'top_20']
        
        # Extract data
        data = {
            'top_10': [summary[m]['top_10']['hit_rate'] * 100 for m in models],
            'top_20': [summary[m]['top_20']['hit_rate'] * 100 for m in models]
        }
        
        # Set up bars
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, data['top_10'], width, 
                      label='Top-10', color='#667eea', alpha=0.8)
        bars2 = ax.bar(x + width/2, data['top_20'], width,
                      label='Top-20', color='#764ba2', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hit Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Ground Truth Validation: Hit Rate Comparison\n(Higher is Better)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([self.model_labels[m] for m in models], fontsize=11)
        ax.legend(fontsize=11, loc='upper left')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add description
        fig.text(0.5, 0.02, 
                'Hit Rate: Percentage of test cases where the true journal appears in Top-K recommendations',
                ha='center', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Hit rate comparison saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_mrr_comparison(self, summary: Dict, output_path: str = None):
        """
        Create grouped bar chart comparing Mean Reciprocal Rank.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        models = ['hybrid', 'tfidf', 'bert']
        
        # Extract data
        data = {
            'top_10': [summary[m]['top_10']['mrr'] for m in models],
            'top_20': [summary[m]['top_20']['mrr'] for m in models]
        }
        
        # Set up bars
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, data['top_10'], width,
                      label='Top-10', color='#4facfe', alpha=0.8)
        bars2 = ax.bar(x + width/2, data['top_20'], width,
                      label='Top-20', color='#00c6ff', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Reciprocal Rank (MRR)', fontsize=12, fontweight='bold')
        ax.set_title('Ground Truth Validation: MRR Comparison\n(Higher is Better)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([self.model_labels[m] for m in models], fontsize=11)
        ax.legend(fontsize=11, loc='upper left')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add description
        fig.text(0.5, 0.02,
                'MRR: Average of 1/rank for each true journal. Rank 1 = 1.0, Rank 2 = 0.5, Rank 10 = 0.1',
                ha='center', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ MRR comparison saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_avg_rank_comparison(self, summary: Dict, output_path: str = None):
        """
        Create grouped bar chart comparing average ranks.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        models = ['hybrid', 'tfidf', 'bert']
        
        # Extract data
        data = {
            'top_10': [summary[m]['top_10']['avg_rank'] for m in models],
            'top_20': [summary[m]['top_20']['avg_rank'] for m in models]
        }
        
        # Set up bars
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, data['top_10'], width,
                      label='Top-10', color='#f093fb', alpha=0.8)
        bars2 = ax.bar(x + width/2, data['top_20'], width,
                      label='Top-20', color='#f5576c', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Rank', fontsize=12, fontweight='bold')
        ax.set_title('Ground Truth Validation: Average Rank Comparison\n(Lower is Better)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([self.model_labels[m] for m in models], fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.invert_yaxis()  # Lower is better, so invert y-axis
        ax.grid(axis='y', alpha=0.3)
        
        # Add description
        fig.text(0.5, 0.02,
                'Average Rank: Mean position of true journal in recommendations (lower is better, Rank 1 is best)',
                ha='center', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Average rank comparison saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_comprehensive_dashboard(self, summary: Dict, output_path: str = None):
        """
        Create a comprehensive 2x2 dashboard with all metrics.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        models = ['hybrid', 'tfidf', 'bert']
        model_names = [self.model_labels[m] for m in models]
        
        # 1. Hit Rate @ Top-10
        ax1 = fig.add_subplot(gs[0, 0])
        data_10 = [summary[m]['top_10']['hit_rate'] * 100 for m in models]
        bars = ax1.bar(model_names, data_10, color=[self.colors[m] for m in models], alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax1.set_ylabel('Hit Rate (%)', fontweight='bold')
        ax1.set_title('Hit Rate @ Top-10', fontweight='bold', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Hit Rate @ Top-20
        ax2 = fig.add_subplot(gs[0, 1])
        data_20 = [summary[m]['top_20']['hit_rate'] * 100 for m in models]
        bars = ax2.bar(model_names, data_20, color=[self.colors[m] for m in models], alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax2.set_ylabel('Hit Rate (%)', fontweight='bold')
        ax2.set_title('Hit Rate @ Top-20', fontweight='bold', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. MRR Comparison
        ax3 = fig.add_subplot(gs[1, 0])
        mrr_10 = [summary[m]['top_10']['mrr'] for m in models]
        mrr_20 = [summary[m]['top_20']['mrr'] for m in models]
        x = np.arange(len(models))
        width = 0.35
        ax3.bar(x - width/2, mrr_10, width, label='Top-10', alpha=0.8, color='#4facfe')
        ax3.bar(x + width/2, mrr_20, width, label='Top-20', alpha=0.8, color='#00c6ff')
        ax3.set_ylabel('MRR', fontweight='bold')
        ax3.set_title('Mean Reciprocal Rank', fontweight='bold', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.set_ylim(0, 1.0)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Average Rank Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        rank_10 = [summary[m]['top_10']['avg_rank'] for m in models]
        rank_20 = [summary[m]['top_20']['avg_rank'] for m in models]
        ax4.bar(x - width/2, rank_10, width, label='Top-10', alpha=0.8, color='#f093fb')
        ax4.bar(x + width/2, rank_20, width, label='Top-20', alpha=0.8, color='#f5576c')
        ax4.set_ylabel('Average Rank', fontweight='bold')
        ax4.set_title('Average Rank (Lower is Better)', fontweight='bold', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names)
        ax4.legend()
        ax4.invert_yaxis()
        ax4.grid(axis='y', alpha=0.3)
        
        # Overall title
        fig.suptitle('Ground Truth Validation: Comprehensive Model Comparison',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Footer
        n_samples = summary['hybrid']['top_10']['n_samples']
        fig.text(0.5, 0.02,
                f'Based on {n_samples} test cases | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                ha='center', fontsize=10, style='italic', color='gray')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comprehensive dashboard saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all(self, summary: Dict, output_dir: str = 'metrics/output/visualizations/ground_truth'):
        """Generate all ground truth validation visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n📊 Generating ground truth validation visualizations...")
        
        # Individual plots
        self.plot_hit_rate_comparison(summary, str(output_path / 'hit_rate_comparison.png'))
        self.plot_mrr_comparison(summary, str(output_path / 'mrr_comparison.png'))
        self.plot_avg_rank_comparison(summary, str(output_path / 'avg_rank_comparison.png'))
        
        # Comprehensive dashboard
        self.plot_comprehensive_dashboard(summary, str(output_path / 'comprehensive_comparison.png'))
        
        print(f"\n✓ All visualizations saved to: {output_path}")


def visualize_from_json(json_path: str = 'metrics/output/ground_truth_results.json'):
    """Load results from JSON and create visualizations."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    visualizer = GroundTruthVisualizer()
    visualizer.plot_all(summary)


if __name__ == '__main__':
    # Example: Load and visualize existing results
    visualize_from_json()
