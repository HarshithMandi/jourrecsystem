"""
User Behavior Visualization Module

Creates visualizations for user metrics:
- Query patterns over time
- Popular journals and topics
- User interaction heatmaps
- Topic trend analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
from pathlib import Path
from collections import Counter


class UserVisualizer:
    """Generate user behavior metric visualizations"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        self.figure_size = (12, 6)
    
    def plot_query_patterns(self, metrics: Dict, output_path: str):
        """
        Plot query patterns over time
        
        Args:
            metrics: User metrics dictionary
            output_path: Path to save figure
        """
        query_patterns = metrics.get('query_patterns', {})
        
        if 'error' in query_patterns:
            print("No query pattern data available")
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Hourly distribution
        ax1 = fig.add_subplot(gs[0, :])
        hourly_dist = query_patterns.get('hourly_distribution', {})
        
        if hourly_dist:
            hours = list(hourly_dist.keys())
            counts = list(hourly_dist.values())
            
            ax1.plot(hours, counts, marker='o', linewidth=2, markersize=6, color='steelblue')
            ax1.fill_between(hours, counts, alpha=0.3, color='steelblue')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Number of Queries')
            ax1.set_title('Query Distribution by Hour')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(0, 24, 2))
            
            # Highlight peak hour
            peak_hour_info = query_patterns.get('peak_hour', {})
            if peak_hour_info:
                peak_hour = peak_hour_info.get('hour', 0)
                ax1.axvline(peak_hour, color='red', linestyle='--', alpha=0.5, label='Peak Hour')
                ax1.legend()
        
        # 2. Daily distribution
        ax2 = fig.add_subplot(gs[1, :])
        daily_dist = query_patterns.get('daily_distribution', {})
        
        if daily_dist:
            days = list(daily_dist.keys())
            counts = list(daily_dist.values())
            
            ax2.bar(range(len(days)), counts, color='coral', alpha=0.7)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Queries')
            ax2.set_title('Query Distribution by Day')
            ax2.set_xticks(range(len(days)))
            ax2.set_xticklabels(days, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Query length statistics
        ax3 = fig.add_subplot(gs[2, 0])
        length_stats = query_patterns.get('query_length_stats', {})
        
        if length_stats:
            stat_names = ['Avg', 'Median', 'Min', 'Max']
            values = [
                length_stats.get('avg_length', 0),
                length_stats.get('median_length', 0),
                length_stats.get('min_length', 0),
                length_stats.get('max_length', 0)
            ]
            
            ax3.bar(stat_names, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
            ax3.set_ylabel('Characters')
            ax3.set_title('Query Length Statistics')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary info
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        
        summary_text = f"""
        Query Pattern Summary:
        
        Total Queries: {query_patterns.get('total_queries', 0)}
        Time Window: {query_patterns.get('time_window_hours', 0)} hours
        Peak Hour: {peak_hour_info.get('hour', 0)}:00
        Peak Count: {peak_hour_info.get('query_count', 0)}
        Avg Query Length: {length_stats.get('avg_length', 0):.0f} chars
        """
        
        ax4.text(0.5, 0.5, summary_text, fontsize=10, family='monospace',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Query patterns saved to: {output_path}")
    
    def plot_popular_journals(self, metrics: Dict, output_path: str):
        """
        Plot most popular recommended journals
        
        Args:
            metrics: User metrics dictionary
            output_path: Path to save figure
        """
        popular = metrics.get('popular_journals', {})
        top_journals = popular.get('top_journals', [])
        
        if not top_journals:
            print("No popular journal data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Top journals by recommendation count
        journals = [j['journal_name'][:30] + '...' if len(j['journal_name']) > 30 
                   else j['journal_name'] for j in top_journals[:10]]
        counts = [j['recommendation_count'] for j in top_journals[:10]]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(journals)))
        ax1.barh(range(len(journals)), counts, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(journals)))
        ax1.set_yticklabels(journals, fontsize=9)
        ax1.set_xlabel('Recommendation Count')
        ax1.set_title('Top 10 Most Recommended Journals')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Average similarity scores
        journals_sim = [j['journal_name'][:30] + '...' if len(j['journal_name']) > 30 
                       else j['journal_name'] for j in top_journals[:10]]
        sim_scores = [j['avg_similarity_score'] for j in top_journals[:10]]
        
        colors = plt.cm.RdYlGn(np.array(sim_scores))
        ax2.barh(range(len(journals_sim)), sim_scores, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(journals_sim)))
        ax2.set_yticklabels(journals_sim, fontsize=9)
        ax2.set_xlabel('Average Similarity Score')
        ax2.set_title('Top 10 Journals by Avg Similarity')
        ax2.invert_yaxis()
        ax2.set_xlim([0, 1.0])
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Popular journals saved to: {output_path}")
    
    def plot_topic_trends(self, metrics: Dict, output_path: str):
        """
        Plot trending topics and keywords
        
        Args:
            metrics: User metrics dictionary
            output_path: Path to save figure
        """
        topic_trends = metrics.get('topic_trends', {})
        
        if 'error' in topic_trends:
            print("No topic trend data available")
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 1, hspace=0.3)
        
        # 1. Top keywords
        ax1 = fig.add_subplot(gs[0, :])
        top_keywords = topic_trends.get('top_keywords', {})
        
        if top_keywords:
            keywords = list(top_keywords.keys())[:20]
            frequencies = list(top_keywords.values())[:20]
            
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(keywords)))
            ax1.bar(range(len(keywords)), frequencies, color=colors, alpha=0.7)
            ax1.set_xlabel('Keywords')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Top 20 Keywords in User Queries')
            ax1.set_xticks(range(len(keywords)))
            ax1.set_xticklabels(keywords, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Top subjects
        ax2 = fig.add_subplot(gs[1, :])
        top_subjects = topic_trends.get('top_subjects', {})
        
        if top_subjects:
            subjects = list(top_subjects.keys())[:15]
            frequencies = list(top_subjects.values())[:15]
            
            colors = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(subjects)))
            ax2.barh(range(len(subjects)), frequencies, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(subjects)))
            ax2.set_yticklabels(subjects, fontsize=9)
            ax2.set_xlabel('Frequency')
            ax2.set_title('Top 15 Subject Areas')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Topic trends saved to: {output_path}")
    
    def plot_interaction_patterns(self, metrics: Dict, output_path: str):
        """
        Plot user interaction patterns
        
        Args:
            metrics: User metrics dictionary
            output_path: Path to save figure
        """
        interactions = metrics.get('interaction_patterns', {})
        
        if 'error' in interactions:
            print("No interaction pattern data available")
            return
        
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Session statistics
        ax1 = fig.add_subplot(gs[0, 0])
        session_stats = interactions.get('session_stats', {})
        
        if session_stats:
            stat_names = ['Avg Queries/Session', 'Median Queries/Session', 'Max Queries/Session']
            values = [
                session_stats.get('avg_queries_per_session', 0),
                session_stats.get('median_queries_per_session', 0),
                session_stats.get('max_queries_per_session', 0)
            ]
            
            ax1.bar(range(len(stat_names)), values, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
            ax1.set_xticks(range(len(stat_names)))
            ax1.set_xticklabels(stat_names, rotation=45, ha='right')
            ax1.set_ylabel('Count')
            ax1.set_title('Session Query Statistics')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. User retention
        ax2 = fig.add_subplot(gs[0, 1])
        retention = interactions.get('user_retention', {})
        
        if retention:
            multi_query = retention.get('multi_query_sessions', 0)
            single_query = interactions.get('total_sessions', 0) - multi_query
            
            sizes = [multi_query, single_query]
            labels = ['Multi-Query\nSessions', 'Single-Query\nSessions']
            colors = ['#2ecc71', '#95a5a6']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})
            ax2.set_title('Session Engagement')
        
        # 3. Recommendation statistics
        ax3 = fig.add_subplot(gs[1, 0])
        rec_stats = interactions.get('recommendation_stats', {})
        
        if rec_stats:
            ax3.bar(['Avg Recs/Query', 'Median Recs/Query'],
                   [rec_stats.get('avg_recommendations_per_query', 0),
                    rec_stats.get('median_recommendations_per_query', 0)],
                   color=['#f39c12', '#9b59b6'], alpha=0.7)
            ax3.set_ylabel('Count')
            ax3.set_title('Recommendations per Query')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        summary_text = f"""
        Interaction Summary:
        
        Total Sessions: {interactions.get('total_sessions', 0)}
        Total Queries: {interactions.get('total_queries', 0)}
        
        Multi-Query Sessions: {retention.get('multi_query_sessions', 0)}
        Retention Rate: {retention.get('multi_query_session_percentage', 0):.1f}%
        
        Avg Queries/Session: {session_stats.get('avg_queries_per_session', 0):.2f}
        Avg Recs/Query: {rec_stats.get('avg_recommendations_per_query', 0):.2f}
        """
        
        ax4.text(0.5, 0.5, summary_text, fontsize=9, family='monospace',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Interaction patterns saved to: {output_path}")
    
    def plot_open_access_preference(self, metrics: Dict, output_path: str):
        """
        Plot open access preference analysis
        
        Args:
            metrics: User metrics dictionary
            output_path: Path to save figure
        """
        oa_pref = metrics.get('open_access_preference', {})
        
        if 'error' in oa_pref:
            print("No open access preference data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Recommendation rate vs availability rate
        oa_rec_rate = oa_pref.get('open_access_recommendation_rate', 0)
        oa_avail_rate = oa_pref.get('open_access_availability_rate', 0)
        
        categories = ['OA Recommendation\nRate', 'OA Availability\nRate']
        values = [oa_rec_rate, oa_avail_rate]
        colors = ['#2ecc71', '#3498db']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Open Access: Recommendations vs Availability')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Counts
        oa_count = oa_pref.get('oa_recommended', 0)
        total_count = oa_pref.get('total_recommended', 0)
        trad_count = total_count - oa_count
        
        sizes = [oa_count, trad_count]
        labels = [f'Open Access\n({oa_count})', f'Traditional\n({trad_count})']
        colors = ['#2ecc71', '#95a5a6']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Recommendation Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Open access preference saved to: {output_path}")
    
    def generate_all_user_plots(self, metrics: Dict, output_dir: str):
        """
        Generate all user behavior visualizations
        
        Args:
            metrics: User metrics dictionary
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating user behavior visualizations...")
        
        self.plot_query_patterns(metrics, str(output_path / "query_patterns.png"))
        self.plot_popular_journals(metrics, str(output_path / "popular_journals.png"))
        self.plot_topic_trends(metrics, str(output_path / "topic_trends.png"))
        self.plot_interaction_patterns(metrics, str(output_path / "interaction_patterns.png"))
        self.plot_open_access_preference(metrics, str(output_path / "open_access_preference.png"))
        
        print(f"All user behavior visualizations saved to: {output_dir}")
