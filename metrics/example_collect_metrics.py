"""
Example: Collect and Export All Metrics

This script demonstrates how to collect all metrics and export them to files.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.metrics_collector import MetricsCollector


def main():
    print("=" * 60)
    print("JOURNAL RECOMMENDATION SYSTEM - METRICS COLLECTION")
    print("=" * 60)
    print()
    
    # Initialize collector
    collector = MetricsCollector()
    
    # Set time window (in hours)
    hours = 24
    
    print(f"Collecting metrics for the last {hours} hours...")
    print()
    
    # Collect all metrics
    all_metrics = collector.collect_all_metrics(hours=hours)
    
    # Export to JSON
    json_output = "metrics/output/all_metrics.json"
    collector.export_all_metrics(json_output, hours=hours)
    print(f"✓ Full metrics exported to: {json_output}")
    
    # Export summary report
    summary_output = "metrics/output/summary_report.txt"
    collector.export_summary_report(summary_output)
    print(f"✓ Summary report exported to: {summary_output}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    
    summary = collector.get_summary_dashboard()
    print(f"System Status: {summary['system_status']}")
    print(f"Health Score: {summary['health_score']:.1f}/100")
    print(f"Total Journals: {summary['total_journals']:,}")
    print(f"Vector Coverage: {summary['vector_coverage']:.1f}%")
    print(f"Queries (24h): {summary['queries_24h']}")
    print(f"Avg Response Time: {summary['avg_response_time_ms']:.2f}ms")
    print(f"P95 Response Time: {summary['p95_response_time_ms']:.2f}ms")
    
    print("\n" + "=" * 60)
    print("METRICS COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
