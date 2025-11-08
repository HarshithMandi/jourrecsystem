"""
Quick Setup Check

Verifies that all required dependencies are installed and the metrics module is ready to use.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if all required packages are installed"""
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    print()
    
    required_packages = {
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'numpy': 'numpy',
        'sqlalchemy': 'sqlalchemy'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - NOT INSTALLED")
            missing_packages.append(package_name)
    
    print()
    
    if missing_packages:
        print("⚠️  Missing packages detected!")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✓ All dependencies installed!")
        return True

def check_database():
    """Check if database is accessible"""
    print("\n" + "=" * 60)
    print("CHECKING DATABASE")
    print("=" * 60)
    print()
    
    try:
        from app.models.base import SessionLocal
        from app.models.entities import Journal
        
        db = SessionLocal()
        journal_count = db.query(Journal).count()
        db.close()
        
        print(f"✓ Database accessible")
        print(f"✓ Journals in database: {journal_count:,}")
        
        if journal_count == 0:
            print("⚠️  Warning: No journals in database. Metrics will be limited.")
            print("   Run data ingestion scripts to populate the database.")
            return False
        elif journal_count < 100:
            print("⚠️  Warning: Low journal count. Consider adding more data.")
            return True
        else:
            print("✓ Database has sufficient data")
            return True
            
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("   Make sure the database is initialized and accessible.")
        return False

def check_metrics_module():
    """Check if metrics module is accessible"""
    print("\n" + "=" * 60)
    print("CHECKING METRICS MODULE")
    print("=" * 60)
    print()
    
    try:
        from metrics.metrics_collector import MetricsCollector
        from metrics.performance_metrics import PerformanceMetrics
        from metrics.accuracy_metrics import AccuracyMetrics
        from metrics.system_metrics import SystemMetrics
        from metrics.user_metrics import UserMetrics
        
        print("✓ Core metrics modules accessible")
        
        from metrics.visualizations.performance_visualizer import PerformanceVisualizer
        from metrics.visualizations.accuracy_visualizer import AccuracyVisualizer
        from metrics.visualizations.system_visualizer import SystemVisualizer
        from metrics.visualizations.user_visualizer import UserVisualizer
        from metrics.visualizations.dashboard_generator import DashboardGenerator
        
        print("✓ Visualization modules accessible")
        print("✓ Metrics module ready to use!")
        return True
        
    except ImportError as e:
        print(f"✗ Metrics module import failed: {e}")
        print("   Make sure you're running from the project root directory.")
        return False

def check_output_directory():
    """Check if output directory exists or can be created"""
    print("\n" + "=" * 60)
    print("CHECKING OUTPUT DIRECTORY")
    print("=" * 60)
    print()
    
    try:
        output_dir = Path("metrics/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directory ready: {output_dir}")
        
        # Check write permissions
        test_file = output_dir / ".test"
        test_file.touch()
        test_file.unlink()
        
        print("✓ Write permissions confirmed")
        return True
        
    except Exception as e:
        print(f"✗ Output directory setup failed: {e}")
        return False

def main():
    """Run all checks"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "METRICS MODULE - SETUP CHECK" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Database", check_database),
        ("Metrics Module", check_metrics_module),
        ("Output Directory", check_output_directory)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP CHECK SUMMARY")
    print("=" * 60)
    print()
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou're ready to use the metrics module!")
        print("\nNext steps:")
        print("  1. python metrics/example_generate_dashboard.py")
        print("  2. Open metrics/output/dashboard.html in your browser")
        print("=" * 60)
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nPlease resolve the issues above before using the metrics module.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
