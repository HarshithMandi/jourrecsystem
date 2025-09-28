#!/usr/bin/env python3
"""
Journal Recommender API Usage Examples
=====================================

This script demonstrates how to use the Journal Recommender API
with various types of research abstracts and different endpoints.
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Sample research abstracts for testing
SAMPLE_ABSTRACTS = {
    "machine_learning": """
    This study presents a novel deep learning approach for automated feature extraction
    in high-dimensional datasets. We developed a convolutional neural network architecture
    that combines attention mechanisms with residual connections to improve classification
    accuracy. Our method was evaluated on several benchmark datasets including ImageNet
    and CIFAR-10, achieving state-of-the-art results. The proposed architecture reduces
    computational complexity by 40% while maintaining superior performance compared to
    existing approaches. These findings have significant implications for real-time
    applications in computer vision and autonomous systems.
    """,
    
    "biology": """
    We investigate the molecular mechanisms underlying protein folding in eukaryotic cells
    using advanced cryo-electron microscopy techniques. Our research reveals novel
    structural intermediates in the folding pathway of membrane proteins, particularly
    focusing on G-protein coupled receptors. Through biochemical assays and molecular
    dynamics simulations, we demonstrate that chaperone proteins play a critical role
    in preventing misfolding events. The results provide new insights into protein
    homeostasis and have potential therapeutic applications for neurodegenerative
    diseases associated with protein aggregation.
    """,
    
    "physics": """
    This paper reports on experimental observations of quantum entanglement in a
    solid-state system using superconducting qubits. We demonstrate coherent control
    of two-qubit gates with fidelities exceeding 99.5% through optimized pulse sequences
    and error correction protocols. Our quantum processor implements a novel topological
    approach to fault-tolerant quantum computation, showing resilience to environmental
    decoherence. The experimental results validate theoretical predictions and represent
    a significant step toward scalable quantum computing architectures for practical
    applications in cryptography and optimization problems.
    """,
    
    "chemistry": """
    We present the synthesis and characterization of a new class of organometallic
    catalysts for asymmetric hydrogenation reactions. Using density functional theory
    calculations and experimental validation, we optimized ligand structures to achieve
    unprecedented enantioselectivity (>98% ee) in the reduction of substituted alkenes.
    X-ray crystallography reveals the three-dimensional catalyst structure and explains
    the observed stereoselectivity through detailed transition state analysis. This
    methodology opens new possibilities for sustainable chemical manufacturing and
    pharmaceutical synthesis applications.
    """
}

def test_single_recommendation(abstract, field_name, top_k=5):
    """Test single recommendation endpoint"""
    print(f"\n🔬 Testing {field_name} abstract...")
    
    payload = {
        "abstract": abstract.strip(),
        "top_k": top_k
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/api/recommend", json=payload, timeout=30)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! ({response_time:.0f}ms client time)")
            print(f"   Server processing: {data['processing_time_ms']}ms")
            print(f"   Query ID: {data['query_id']}")
            print(f"   Found {len(data['recommendations'])} recommendations")
            
            print(f"\n📋 Top {min(3, len(data['recommendations']))} journals for {field_name}:")
            for rec in data['recommendations'][:3]:
                print(f"   {rec['rank']}. {rec['journal_name']}")
                print(f"      Similarity: {rec['similarity_score']:.4f}")
            
            return data
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None

def test_batch_recommendation():
    """Test batch recommendation endpoint"""
    print("\n🔄 Testing batch recommendations...")
    
    # Use first 3 abstracts for batch test
    abstracts_list = list(SAMPLE_ABSTRACTS.values())[:3]
    
    payload = {
        "abstracts": [abs.strip() for abs in abstracts_list],
        "top_k": 3
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/api/batch-recommend", json=payload, timeout=60)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch request successful! ({response_time:.0f}ms total)")
            print(f"   Server processing: {data['total_processing_time_ms']}ms")
            print(f"   Processed {len(data['results'])} abstracts")
            
            for i, result in enumerate(data['results']):
                field_name = list(SAMPLE_ABSTRACTS.keys())[i]
                print(f"\n   📊 {field_name.replace('_', ' ').title()}:")
                if result['recommendations']:
                    top_rec = result['recommendations'][0]
                    print(f"      Top journal: {top_rec['journal_name']}")
                    print(f"      Similarity: {top_rec['similarity_score']:.4f}")
                else:
                    print("      No recommendations found")
            
            return data
        else:
            print(f"❌ Batch request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Batch request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch request error: {e}")
        return None

def test_stats_endpoint():
    """Test database statistics endpoint"""
    print("\n📊 Testing statistics endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Statistics retrieved successfully!")
            print(f"   📚 Total journals: {data['total_journals']}")
            print(f"   🔍 Total queries: {data['total_queries']}")
            print(f"   📈 Total recommendations: {data['total_recommendations']}")
            print(f"   🧠 Journals with ML profiles: {data['journals_with_profiles']}")
            if data['avg_similarity_score']:
                print(f"   ⭐ Average similarity score: {data['avg_similarity_score']:.3f}")
            else:
                print("   ⭐ No similarity data available yet")
            
            return data
        else:
            print(f"❌ Stats request failed: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Stats request error: {e}")
        return None

def test_health_check():
    """Test API health check"""
    print("\n🏥 Testing health check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/ping", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy!")
            print(f"   Status: {data['status']}")
            print(f"   Service: {data['service']}")
            print(f"   Version: {data['version']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Run comprehensive API examples"""
    print("🧪 Journal Recommender API - Usage Examples")
    print("=" * 50)
    print(f"🌐 API Base URL: {API_BASE_URL}")
    
    # Test API health first
    if not test_health_check():
        print("\n❌ API is not available. Make sure the server is running:")
        print("   python launch_api.py")
        print("   or")
        print("   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return
    
    # Test statistics
    stats = test_stats_endpoint()
    
    if stats and stats['total_journals'] == 0:
        print("\n⚠️  No journals found in database. Run data ingestion first:")
        print("   python scripts/ingest_openalex.py")
        return
    
    # Test individual recommendations
    print("\n" + "="*50)
    print("🔬 Individual Recommendation Tests")
    print("="*50)
    
    results = {}
    for field, abstract in SAMPLE_ABSTRACTS.items():
        result = test_single_recommendation(abstract, field, top_k=5)
        if result:
            results[field] = result
        time.sleep(0.5)  # Small delay between requests
    
    # Test batch recommendations
    print("\n" + "="*50)
    print("🔄 Batch Recommendation Test")
    print("="*50)
    
    batch_result = test_batch_recommendation()
    
    # Summary
    print("\n" + "="*50)
    print("📋 Summary")
    print("="*50)
    
    print(f"✅ Successful individual tests: {len(results)}/{len(SAMPLE_ABSTRACTS)}")
    print(f"✅ Batch test: {'Success' if batch_result else 'Failed'}")
    
    if results:
        print("\n🏆 Best matches by field:")
        for field, result in results.items():
            if result['recommendations']:
                top = result['recommendations'][0]
                print(f"   {field.replace('_', ' ').title()}: {top['journal_name']}")
                print(f"      (score: {top['similarity_score']:.4f})")
    
    print(f"\n🌐 Interactive API Documentation: {API_BASE_URL}/docs")
    print("🎯 Use the web interface for more detailed testing!")

if __name__ == "__main__":
    main()