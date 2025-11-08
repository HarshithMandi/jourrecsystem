"""
Example: Generate All Visualizations

This script demonstrates how to generate all visualization plots.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.metrics_collector import MetricsCollector
from metrics.visualizations.performance_visualizer import PerformanceVisualizer
from metrics.visualizations.accuracy_visualizer import AccuracyVisualizer
from metrics.visualizations.system_visualizer import SystemVisualizer
from metrics.visualizations.user_visualizer import UserVisualizer


def main():
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    print()
    
    # Initialize collector and visualizers
    collector = MetricsCollector()
    perf_viz = PerformanceVisualizer()
    acc_viz = AccuracyVisualizer()
    sys_viz = SystemVisualizer()
    user_viz = UserVisualizer()
    
    # Set time window
    hours = 24
    
    print(f"Collecting metrics for the last {hours} hours...")
    
    # Collect metrics
    performance_metrics = collector.get_performance_metrics(hours)
    accuracy_metrics = collector.get_accuracy_metrics(hours)
    system_metrics = collector.get_system_metrics()
    user_metrics = collector.get_user_metrics(hours)
    
    # Create output directories
    output_base = Path("metrics/output/visualizations")
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    print()
    
    # Generate all visualizations
    print("1. Performance Visualizations...")
    perf_viz.generate_all_performance_plots(
        performance_metrics,
        str(output_base / "performance")
    )
    
    print("\n2. Accuracy Visualizations...")
    acc_viz.generate_all_accuracy_plots(
        accuracy_metrics,
        str(output_base / "accuracy")
    )
    
    print("\n3. System Visualizations...")
    sys_viz.generate_all_system_plots(
        system_metrics,
        str(output_base / "system")
    )
    
    print("\n4. User Behavior Visualizations...")
    user_viz.generate_all_user_plots(
        user_metrics,
        str(output_base / "user")
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nAll visualizations saved to: {output_base}")


if __name__ == "__main__":
    main()
