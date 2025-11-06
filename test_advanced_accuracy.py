#!/usr/bin/env python3
"""
Advanced System Accuracy Test
Test the advanced recommendation system specifically
"""

import sys
from pathlib import Path
import requests
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import time
import random

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# API Configuration
API_BASE_URL = "http://localhost:8000"

class AdvancedAccuracyTester:
    """Test advanced recommendation accuracy"""
    
    def __init__(self, db_path: str = "data/journal_rec.db"):
        self.db_path = db_path
        self.api_available = self.check_api()
        
    def check_api(self) -> bool:
        """Check if API server is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/ping", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_test_papers(self, limit: int = 25) -> List[Dict]:
        """Get papers with abstracts from database for testing"""
        
        query = """
        SELECT 
            w.id,
            w.title,
            w.abstract,
            w.journal_id,
            j.name as journal_name,
            j.publisher
        FROM works w
        JOIN journals j ON w.journal_id = j.id
        WHERE w.abstract IS NOT NULL 
        AND LENGTH(w.abstract) > 100
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()
            
            papers = df.to_dict('records')
            print(f"✅ Found {len(papers)} papers with abstracts for testing")
            return papers
            
        except Exception as e:
            print(f"❌ Error fetching test papers: {e}")
            return []
    
    def get_advanced_recommendations(self, abstract: str, top_k: int = 10) -> List[Dict]:
        """Get recommendations from advanced API"""
        
        if not self.api_available:
            print("❌ API server not available")
            return []
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/recommend",  # Uses advanced system now
                json={"abstract": abstract, "top_k": top_k},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("recommendations", [])
            else:
                print(f"❌ API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return []
    
    def test_single_paper(self, paper: Dict, top_k: int = 10) -> Dict:
        """Test a single paper and return results"""
        
        abstract = paper['abstract']
        actual_journal = paper['journal_name']
        
        # Get recommendations
        recommendations = self.get_advanced_recommendations(abstract, top_k)
        
        if not recommendations:
            return {
                'paper_id': paper['id'],
                'title': paper['title'][:100] + '...',
                'actual_journal': actual_journal,
                'recommendations_found': False,
                'hit_at_1': False,
                'hit_at_3': False,
                'hit_at_5': False,
                'hit_at_10': False,
                'position': 0,
                'mrr': 0,
                'top_similarity_score': 0,
                'error': 'No recommendations returned'
            }
        
        # Find position of actual journal in recommendations
        position = 0
        recommended_journals = [rec['journal_name'] for rec in recommendations]
        
        for i, recommended_journal in enumerate(recommended_journals, 1):
            if recommended_journal.lower() == actual_journal.lower():
                position = i
                break
        
        # Calculate metrics
        hit_at_1 = position == 1
        hit_at_3 = 1 <= position <= 3
        hit_at_5 = 1 <= position <= 5
        hit_at_10 = 1 <= position <= 10
        mrr = 1.0 / position if position > 0 else 0
        
        top_similarity_score = recommendations[0]['similarity_score'] if recommendations else 0
        
        return {
            'paper_id': paper['id'],
            'title': paper['title'][:100] + '...',
            'actual_journal': actual_journal,
            'recommended_journals': recommended_journals[:5],  # Top 5 for display
            'recommendations_found': True,
            'hit_at_1': hit_at_1,
            'hit_at_3': hit_at_3,
            'hit_at_5': hit_at_5,
            'hit_at_10': hit_at_10,
            'position': position,
            'mrr': mrr,
            'top_similarity_score': top_similarity_score,
            'total_recommendations': len(recommendations)
        }
    
    def calculate_accuracy_metrics(self, test_results: List[Dict]) -> Dict:
        """Calculate various accuracy metrics"""
        
        total_tests = len(test_results)
        if total_tests == 0:
            return {}
        
        # Hit rate at different K values
        hit_at_1 = sum(1 for r in test_results if r['hit_at_1']) / total_tests
        hit_at_3 = sum(1 for r in test_results if r['hit_at_3']) / total_tests
        hit_at_5 = sum(1 for r in test_results if r['hit_at_5']) / total_tests
        hit_at_10 = sum(1 for r in test_results if r['hit_at_10']) / total_tests
        
        # Mean reciprocal rank
        mrr_scores = [r['mrr'] for r in test_results if r['mrr'] > 0]
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
        
        # Average position of correct journal (when found)
        positions = [r['position'] for r in test_results if r['position'] > 0]
        avg_position = sum(positions) / len(positions) if positions else 0
        
        # Top score analysis
        avg_similarity = sum(r['top_similarity_score'] for r in test_results) / total_tests
        
        return {
            'total_tests': total_tests,
            'hit_rate_at_1': hit_at_1,
            'hit_rate_at_3': hit_at_3,
            'hit_rate_at_5': hit_at_5,
            'hit_rate_at_10': hit_at_10,
            'mean_reciprocal_rank': avg_mrr,
            'avg_position_when_found': avg_position,
            'avg_top_similarity_score': avg_similarity,
            'coverage': len(mrr_scores) / total_tests  # Fraction of papers where journal was found
        }
    
    def run_advanced_accuracy_test(self, num_papers: int = 25, top_k: int = 10) -> Dict:
        """Run complete advanced accuracy test"""
        
        print("🚀 Advanced Recommendation System Accuracy Test")
        print("=" * 50)
        
        if not self.api_available:
            print("❌ API server not running. Please start it first:")
            print("   uvicorn app.main:app --reload --port 8000")
            return {}
        
        print("✅ API server is running")
        
        # Get test papers
        print(f"\n📚 Fetching {num_papers} test papers...")
        test_papers = self.get_test_papers(num_papers)
        
        if not test_papers:
            print("❌ No test papers found")
            return {}
        
        print(f"✅ Found {len(test_papers)} papers for testing")
        
        # Run tests
        print(f"\n🔍 Testing advanced recommendations (top-{top_k})...")
        test_results = []
        
        for i, paper in enumerate(test_papers, 1):
            print(f"   Testing paper {i}/{len(test_papers)}: {paper['title'][:50]}...")
            
            result = self.test_single_paper(paper, top_k)
            test_results.append(result)
            
            # Small delay to be nice to the API
            time.sleep(0.1)
        
        # Calculate metrics
        print("\n📊 Calculating accuracy metrics...")
        metrics = self.calculate_accuracy_metrics(test_results)
        
        # Generate report
        report = self.generate_accuracy_report(metrics, test_results)
        
        return {
            'metrics': metrics,
            'individual_results': test_results,
            'report': report
        }
    
    def generate_accuracy_report(self, metrics: Dict, test_results: List[Dict]) -> str:
        """Generate accuracy report"""
        
        report = f"""
# Advanced Recommendation System Accuracy Report

## Summary Metrics
- **Total Papers Tested**: {metrics['total_tests']}
- **Hit Rate @ 1**: {metrics['hit_rate_at_1']:.1%} (Exact match in top recommendation)
- **Hit Rate @ 3**: {metrics['hit_rate_at_3']:.1%} (Correct journal in top 3)
- **Hit Rate @ 5**: {metrics['hit_rate_at_5']:.1%} (Correct journal in top 5)
- **Hit Rate @ 10**: {metrics['hit_rate_at_10']:.1%} (Correct journal in top 10)
- **Mean Reciprocal Rank**: {metrics['mean_reciprocal_rank']:.3f}
- **Coverage**: {metrics['coverage']:.1%} (Papers where correct journal was found)
- **Avg Position When Found**: {metrics['avg_position_when_found']:.1f}
- **Avg Top Similarity Score**: {metrics['avg_top_similarity_score']:.3f}

## Performance Assessment
"""
        
        # Performance assessment
        if metrics['hit_rate_at_5'] >= 0.4:
            report += "🎯 **Excellent Performance** - Advanced system working very well!\n"
        elif metrics['hit_rate_at_5'] >= 0.25:
            report += "✅ **Good Performance** - Significant improvement achieved\n"
        elif metrics['hit_rate_at_5'] >= 0.15:
            report += "⚠️ **Moderate Performance** - Some improvement, needs more work\n"
        else:
            report += "❌ **Needs More Work** - Limited improvement\n"
        
        if metrics['mean_reciprocal_rank'] >= 0.3:
            report += "🎯 **Excellent Ranking Quality** - Correct journals ranked highly\n"
        else:
            report += "⚠️ **Ranking Needs Work** - Correct journals not prioritized well\n"
        
        # Sample results
        report += "\n## Sample Test Results\n"
        successful_tests = [r for r in test_results if r['hit_at_10']][:5]
        failed_tests = [r for r in test_results if not r['hit_at_10']][:3]
        
        if successful_tests:
            report += "\n### ✅ Successful Predictions\n"
            for result in successful_tests:
                report += f"- **{result['title']}**\n"
                report += f"  - Actual: {result['actual_journal']}\n"
                report += f"  - Position: #{result['position']}\n"
                report += f"  - Top recommendations: {', '.join(result['recommended_journals'][:3])}\n\n"
        
        if failed_tests:
            report += "\n### ❌ Failed Predictions\n"
            for result in failed_tests:
                report += f"- **{result['title']}**\n"
                report += f"  - Actual: {result['actual_journal']}\n"
                report += f"  - Top recommendations: {', '.join(result['recommended_journals'][:3])}\n\n"
        
        return report

def main():
    """Main function to run advanced accuracy test"""
    
    tester = AdvancedAccuracyTester()
    
    # Run test with 25 papers
    results = tester.run_advanced_accuracy_test(num_papers=25, top_k=10)
    
    if results:
        # Print report
        print(results['report'])
        
        # Save results
        import json
        from datetime import datetime
        
        output = {
            'test_date': datetime.now().isoformat(),
            'system': 'advanced',
            'metrics': results['metrics'],
            'sample_results': results['individual_results'][:10]  # Save first 10 for review
        }
        
        with open("advanced_accuracy_results.json", 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to advanced_accuracy_results.json")
        print(f"\n🎯 Advanced System: {results['metrics']['hit_rate_at_5']:.1%} accuracy at top-5")
        
        # Compare with baseline
        baseline_accuracy = 0.12  # Original system had 12%
        improvement = results['metrics']['hit_rate_at_5'] - baseline_accuracy
        
        print(f"\n🏆 IMPROVEMENT ANALYSIS:")
        print(f"   Baseline:  {baseline_accuracy:.1%}")
        print(f"   Advanced:  {results['metrics']['hit_rate_at_5']:.1%}")
        print(f"   Improvement: {improvement:+.1%}")
        
        if improvement > 0.1:
            print(f"   🎉 SIGNIFICANT IMPROVEMENT! {improvement:.1%} better!")
        elif improvement > 0.05:
            print(f"   ✅ Good improvement: {improvement:.1%} better")
        elif improvement > 0:
            print(f"   📈 Small improvement: {improvement:.1%} better")
        else:
            print(f"   ⚠️ No significant improvement")
    else:
        print("❌ Test failed - check API server and database")

if __name__ == "__main__":
    main()