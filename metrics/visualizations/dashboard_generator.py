"""
Dashboard Generator

Creates comprehensive HTML dashboards with all visualizations
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict

from ..metrics_collector import MetricsCollector
from .performance_visualizer import PerformanceVisualizer
from .accuracy_visualizer import AccuracyVisualizer
from .system_visualizer import SystemVisualizer
from .user_visualizer import UserVisualizer


class DashboardGenerator:
    """
    Generate comprehensive metrics dashboard with all visualizations
    
    Usage:
        generator = DashboardGenerator()
        generator.generate_full_dashboard('output/dashboard.html')
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.perf_viz = PerformanceVisualizer()
        self.acc_viz = AccuracyVisualizer()
        self.sys_viz = SystemVisualizer()
        self.user_viz = UserVisualizer()
    
    def generate_full_dashboard(self, output_path: str, hours: int = 24):
        """
        Generate complete metrics dashboard
        
        Args:
            output_path: Path to output HTML file
            hours: Time window for metrics
        """
        print(f"Generating comprehensive metrics dashboard...")
        
        # Collect all metrics
        all_metrics = self.collector.collect_all_metrics(hours)
        
        # Create output directory
        output_dir = Path(output_path).parent
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all visualizations
        print("Creating performance visualizations...")
        self.perf_viz.generate_all_performance_plots(
            all_metrics['performance'], 
            str(vis_dir / 'performance')
        )
        
        print("Creating accuracy visualizations...")
        self.acc_viz.generate_all_accuracy_plots(
            all_metrics['accuracy'],
            str(vis_dir / 'accuracy')
        )
        
        print("Creating system visualizations...")
        self.sys_viz.generate_all_system_plots(
            all_metrics['system'],
            str(vis_dir / 'system')
        )
        
        print("Creating user behavior visualizations...")
        self.user_viz.generate_all_user_plots(
            all_metrics['user'],
            str(vis_dir / 'user')
        )
        
        # Generate HTML dashboard
        print("Generating HTML dashboard...")
        html_content = self._create_html_dashboard(all_metrics, vis_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Dashboard generated successfully: {output_path}")
        print(f"✓ Visualizations saved to: {vis_dir}")
        
        return output_path
    
    def _create_html_dashboard(self, metrics: Dict, vis_dir: Path) -> str:
        """Create HTML dashboard content"""
        
        summary = self.collector.get_summary_dashboard()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Journal Recommendation System - Metrics Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .summary-card h3 {{
            color: #667eea;
            font-size: 2em;
            margin: 10px 0;
        }}
        
        .summary-card p {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .status.HEALTHY {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status.WARNING {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status.CRITICAL {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .section {{
            padding: 40px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        
        .visualization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}
        
        .viz-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .viz-item img {{
            width: 100%;
            border-radius: 5px;
            margin-top: 10px;
        }}
        
        .viz-item h3 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Journal Recommendation System</h1>
            <p>Comprehensive Metrics Dashboard</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <p>System Status</p>
                <div class="status {summary['system_status']}">{summary['system_status']}</div>
            </div>
            <div class="summary-card">
                <p>Health Score</p>
                <h3>{summary['health_score']:.1f}%</h3>
            </div>
            <div class="summary-card">
                <p>Total Journals</p>
                <h3>{summary['total_journals']:,}</h3>
            </div>
            <div class="summary-card">
                <p>Vector Coverage</p>
                <h3>{summary['vector_coverage']:.1f}%</h3>
            </div>
            <div class="summary-card">
                <p>Queries (24h)</p>
                <h3>{summary['queries_24h']}</h3>
            </div>
            <div class="summary-card">
                <p>Avg Response Time</p>
                <h3>{summary['avg_response_time_ms']:.1f}ms</h3>
            </div>
        </div>
        
        <div class="section">
            <h2>⚡ Performance Metrics</h2>
            <div class="visualization-grid">
                {self._get_viz_html('performance/response_time_stats.png', 'Response Time Statistics', vis_dir)}
                {self._get_viz_html('performance/response_time_distribution.png', 'Response Time Distribution', vis_dir)}
                {self._get_viz_html('performance/throughput.png', 'System Throughput', vis_dir)}
                {self._get_viz_html('performance/component_breakdown.png', 'Component Execution Breakdown', vis_dir)}
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Accuracy Metrics</h2>
            <div class="visualization-grid">
                {self._get_viz_html('accuracy/similarity_distribution.png', 'Similarity Score Distribution', vis_dir)}
                {self._get_viz_html('accuracy/quality_breakdown.png', 'Recommendation Quality Breakdown', vis_dir)}
                {self._get_viz_html('accuracy/ranking_quality.png', 'Ranking Quality Metrics', vis_dir)}
                {self._get_viz_html('accuracy/diversity_metrics.png', 'Recommendation Diversity', vis_dir)}
            </div>
        </div>
        
        <div class="section">
            <h2>🖥️ System Metrics</h2>
            <div class="visualization-grid">
                {self._get_viz_html('system/database_stats.png', 'Database Statistics', vis_dir)}
                {self._get_viz_html('system/vector_quality.png', 'Vector Quality Analysis', vis_dir)}
                {self._get_viz_html('system/system_health.png', 'System Health Dashboard', vis_dir)}
            </div>
        </div>
        
        <div class="section">
            <h2>👥 User Behavior Metrics</h2>
            <div class="visualization-grid">
                {self._get_viz_html('user/query_patterns.png', 'Query Patterns Over Time', vis_dir)}
                {self._get_viz_html('user/popular_journals.png', 'Most Popular Journals', vis_dir)}
                {self._get_viz_html('user/topic_trends.png', 'Topic Trends & Keywords', vis_dir)}
                {self._get_viz_html('user/interaction_patterns.png', 'User Interaction Patterns', vis_dir)}
                {self._get_viz_html('user/open_access_preference.png', 'Open Access Preference', vis_dir)}
            </div>
        </div>
        
        <div class="footer">
            <p>Journal Recommendation System - Metrics Dashboard v1.0</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Powered by TF-IDF + BERT Hybrid Recommendation Engine
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _get_viz_html(self, rel_path: str, title: str, vis_dir: Path) -> str:
        """Generate HTML for a single visualization"""
        full_path = vis_dir / rel_path
        
        if full_path.exists():
            return f"""
            <div class="viz-item">
                <h3>{title}</h3>
                <img src="visualizations/{rel_path}" alt="{title}">
            </div>
            """
        else:
            return f"""
            <div class="viz-item">
                <h3>{title}</h3>
                <p style="color: #999; padding: 40px; text-align: center;">Visualization not available</p>
            </div>
            """
