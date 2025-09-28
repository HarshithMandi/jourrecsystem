#!/usr/bin/env python3
"""
Test script for the Journal Recommender API
"""

import requests
import json
import time

# API Base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/ping")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Database: {data.get('database', 'N/A')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_recommendation_endpoint():
    """Test the main recommendation endpoint"""
    print("\n🔍 Testing recommendation endpoint...")
    
    # Sample research abstract
    test_abstract = """
    This study presents a novel machine learning approach for predicting protein structures
    using deep neural networks. We developed a transformer-based architecture that incorporates
    evolutionary information and amino acid properties to improve fold prediction accuracy.
    Our method achieved state-of-the-art results on the CASP14 benchmark, demonstrating
    significant improvements over existing approaches. The model was trained on a diverse
    dataset of protein sequences and structures, and we validated our results through
    cross-validation and independent testing. These findings have important implications
    for drug discovery and structural biology research.
    """
    
    payload = {
        "abstract": test_abstract.strip(),
        "top_k": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/recommend", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Recommendation request successful!")
            print(f"   Query ID: {data['query_id']}")
            print(f"   Processing time: {data['processing_time_ms']}ms")
            print(f"   Total journals in DB: {data['total_journals']}")
            print(f"   Recommendations returned: {len(data['recommendations'])}")
            
            print("\n📋 Top recommendations:")
            for rec in data['recommendations'][:3]:
                print(f"   {rec['rank']}. {rec['journal_name']} (score: {rec['similarity_score']:.3f})")
            
            return True
        else:
            print(f"❌ Recommendation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Recommendation error: {e}")
        return False

def test_batch_recommendation():
    """Test the batch recommendation endpoint"""
    print("\n🔍 Testing batch recommendation endpoint...")
    
    abstracts = [
        "We study quantum computing algorithms for optimization problems in machine learning applications.",
        "This paper presents a new approach to natural language processing using transformer architectures.",
        "Our research focuses on climate change impacts on biodiversity in tropical ecosystems."
    ]
    
    payload = {
        "abstracts": abstracts,
        "top_k": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/batch-recommend", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch recommendation successful!")
            print(f"   Total processing time: {data['total_processing_time_ms']}ms")
            print(f"   Results for {len(data['results'])} abstracts")
            
            for i, result in enumerate(data['results']):
                print(f"\n   Abstract {i+1}: {len(result['recommendations'])} recommendations")
                if result['recommendations']:
                    top_rec = result['recommendations'][0]
                    print(f"   Top journal: {top_rec['journal_name']} (score: {top_rec['similarity_score']:.3f})")
            
            return True
        else:
            print(f"❌ Batch recommendation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch recommendation error: {e}")
        return False

def test_stats_endpoint():
    """Test the database statistics endpoint"""
    print("\n🔍 Testing stats endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/stats")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Stats request successful!")
            print(f"   Total journals: {data['total_journals']}")
            print(f"   Total queries: {data['total_queries']}")
            print(f"   Total recommendations: {data['total_recommendations']}")
            print(f"   Journals with ML profiles: {data['journals_with_profiles']}")
            print(f"   Average similarity score: {data['avg_similarity_score']}")
            
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Stats error: {e}")
        return False

def test_api_docs():
    """Test if API documentation is accessible"""
    print("\n🔍 Testing API documentation...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API docs accessible at /docs")
            return True
        else:
            print(f"❌ API docs failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API docs error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting Journal Recommender API Tests")
    print("=" * 50)
    
    tests = [
        test_health_endpoint,
        test_recommendation_endpoint,
        test_batch_recommendation,
        test_stats_endpoint,
        test_api_docs
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Your API is working perfectly.")
        print(f"\n🌐 API Documentation: {BASE_URL}/docs")
        print(f"🔧 API Health Check: {BASE_URL}/ping")
        print(f"📡 Main Endpoint: {BASE_URL}/api/recommend")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
        print("\nMake sure:")
        print("1. The server is running (python -m uvicorn app.main:app --host 0.0.0.0 --port 8000)")
        print("2. The database is initialized with data")
        print("3. ML vectors are built")

if __name__ == "__main__":
    main()