"""
Ground Truth Validation Runner

Runs complete ground truth validation comparing:
- Hybrid (TF-IDF + BERT) - Current system
- TF-IDF Only
- BERT Only

Evaluates on Top-10 and Top-20 recommendations.
Generates comprehensive visualizations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.models.base import SessionLocal
from metrics.ground_truth_evaluator import GroundTruthEvaluator
from metrics.visualizations.ground_truth_visualizer import GroundTruthVisualizer


def main():
    """Run complete ground truth validation pipeline."""
    print("\n" + "="*80)
    print("GROUND TRUTH VALIDATION SYSTEM")
    print("="*80)
    print("📊 Comparing Three Approaches:")
    print("   1. Hybrid (TF-IDF + BERT) - Current System")
    print("   2. TF-IDF Only")
    print("   3. BERT Only")
    print("\n📈 Metrics:")
    print("   - Hit Rate @ K=10, K=20")
    print("   - Mean Reciprocal Rank (MRR)")
    print("   - Average Rank")
    print("="*80 + "\n")
    
    # Get database connection
    db = SessionLocal()
    
    try:
        # Step 1: Initialize evaluator
        print("🔧 Step 1: Initializing evaluator...")
        evaluator = GroundTruthEvaluator(db)
        
        # Step 2: Load or create test set
        print("\n🔧 Step 2: Loading test set...")
        # Try to load targeted test cases first
        test_cases = evaluator.load_from_json('metrics/test_data/targeted_test_cases.json')
        if not test_cases:
            print("   No targeted test data found. Creating synthetic test set...")
            evaluator.create_synthetic_test_set(n_samples=20)  # Reduced to 20 for faster testing
        
        # Step 3: Evaluate all models
        print("\n🔧 Step 3: Evaluating models...")
        print("   This may take a few minutes...\n")
        summary = evaluator.evaluate_all_models(top_k_values=[10, 20])
        
        # Step 4: Print results
        print("\n🔧 Step 4: Displaying results...")
        evaluator.print_results()
        
        # Step 5: Export results
        print("\n🔧 Step 5: Exporting results to JSON...")
        evaluator.export_results()
        
        # Step 6: Generate visualizations
        print("\n🔧 Step 6: Generating visualizations...")
        visualizer = GroundTruthVisualizer()
        visualizer.plot_all(summary)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ VALIDATION COMPLETE!")
        print("="*80)
        print("\n📁 Generated Files:")
        print("   • metrics/output/ground_truth_results.json - Raw results")
        print("   • metrics/output/visualizations/ground_truth/hit_rate_comparison.png")
        print("   • metrics/output/visualizations/ground_truth/mrr_comparison.png")
        print("   • metrics/output/visualizations/ground_truth/avg_rank_comparison.png")
        print("   • metrics/output/visualizations/ground_truth/comprehensive_comparison.png")
        print("\n💡 Open the PNG files to view the comparison charts!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        db.close()


if __name__ == '__main__':
    main()
