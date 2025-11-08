import requests
import json

def test_advanced_features():
    """Test the new advanced analysis features"""
    base_url = "http://localhost:8000"
    test_abstract = "This study presents machine learning approaches for protein structure prediction using deep neural networks and computational biology methods for drug discovery."
    
    print("🧪 Testing Advanced Journal Recommendation Features")
    print("=" * 60)
    
    # Test detailed recommendations
    print("\n1. Testing Detailed Recommendations...")
    try:
        response = requests.post(
            f"{base_url}/api/recommend-detailed",
            json={"abstract": test_abstract, "top_k": 3},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success! Got {len(data['recommendations'])} detailed recommendations")
            print(f"   📊 Processing time: {data['processing_time_ms']:.1f}ms")
            
            # Show first result with all similarity scores
            if data['recommendations']:
                first = data['recommendations'][0]
                print(f"   🥇 Top result: {first['journal_name']}")
                print(f"      Combined: {first['similarity_combined']:.3f}")
                print(f"      TF-IDF: {first['similarity_tfidf']:.3f}")
                print(f"      BERT: {first['similarity_bert']:.3f}")
        else:
            print(f"   ❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test ranking comparisons
    print("\n2. Testing Ranking Comparisons...")
    try:
        response = requests.post(
            f"{base_url}/api/compare-rankings",
            json={"abstract": test_abstract, "top_k": 3},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            comparisons = data['comparisons']
            print(f"   ✅ Success! Got comparison data")
            print(f"   📊 Processing time: {data['processing_time_ms']:.1f}ms")
            
            # Show ranking differences
            methods = ['similarity_ranking', 'tfidf_only_ranking', 'bert_only_ranking', 'impact_factor_ranking']
            for method in methods:
                if method in comparisons and comparisons[method]:
                    first_journal = comparisons[method][0]['journal_name']
                    print(f"   📈 {method.replace('_', ' ').title()}: {first_journal}")
        else:
            print(f"   ❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test text analysis
    print("\n3. Testing Text Analysis...")
    try:
        response = requests.post(
            f"{base_url}/api/analyze-text",
            json={"abstract": test_abstract},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            analysis = data['analysis']
            print(f"   ✅ Success! Text analysis completed")
            print(f"   📊 Processing time: {data['processing_time_ms']:.1f}ms")
            print(f"   📝 Total words: {analysis['total_words']}")
            print(f"   🔤 Unique words: {analysis['unique_words']}")
            print(f"   📏 Avg word length: {analysis['avg_word_length']:.1f}")
            print(f"   📄 Sentences: {analysis['sentence_count']}")
            
            # Show top words
            if analysis['word_frequency']:
                top_words = list(analysis['word_frequency'].keys())[:5]
                print(f"   🏷️  Top words: {', '.join(top_words)}")
        else:
            print(f"   ❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Advanced features testing completed!")
    print("🌐 Dashboard available at: http://localhost:8501")
    print("📊 Try the 'Advanced Analysis' page for interactive visualizations!")

if __name__ == "__main__":
    test_advanced_features()