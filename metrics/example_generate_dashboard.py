"""
Example: Generate Complete Dashboard

This script generates a comprehensive HTML dashboard with all metrics and visualizations.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.visualizations.dashboard_generator import DashboardGenerator


def main():
    print("=" * 70)
    print("JOURNAL RECOMMENDATION SYSTEM - DASHBOARD GENERATION")
    print("=" * 70)
    print()
    
    # Initialize dashboard generator
    generator = DashboardGenerator()
    
    # Set time window (in hours)
    hours = 24
    
    # Output path
    output_path = "metrics/output/dashboard.html"
    
    print(f"Generating dashboard for the last {hours} hours...")
    print("This will collect all metrics and generate all visualizations...")
    print()
    
    # Generate dashboard
    try:
        dashboard_path = generator.generate_full_dashboard(output_path, hours=hours)
        
        print("\n" + "=" * 70)
        print("✓ DASHBOARD GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📊 Dashboard: {dashboard_path}")
        print(f"🖼️  Visualizations: {Path(dashboard_path).parent / 'visualizations'}")
        print(f"\n💡 Open {dashboard_path} in your web browser to view the dashboard")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
