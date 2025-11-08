#!/usr/bin/env python3
"""
Quick test for the API fix
"""
import requests
import json

def test_api_fix():
    """Test if the API data structure mismatch is fixed"""
    url = "http://localhost:8000/api/recommend"
    
    test_data = {
        "abstract": "This study presents machine learning approaches for protein structure prediction using deep neural networks and computational biology methods for advancing structural biology research.",
        "top_k": 5
    }
    
    try:
        print("Testing API endpoint...")
        response = requests.post(url, json=test_data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Request Successful!")
            print(f"Query ID: {data.get('query_id')}")
            print(f"Number of recommendations: {len(data.get('recommendations', []))}")
            print(f"Total journals: {data.get('total_journals')}")
            print(f"Processing time: {data.get('processing_time_ms')} ms")
            
            # Show first recommendation
            if data.get('recommendations'):
                first_rec = data['recommendations'][0]
                print(f"\nFirst recommendation:")
                print(f"  Journal: {first_rec['journal_name']}")
                print(f"  Score: {first_rec['similarity_score']:.4f}")
                print(f"  Rank: {first_rec['rank']}")
            
            return True
        else:
            print("❌ API Request Failed!")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("Make sure the API server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_api_fix()
    if success:
        print("\n🎉 API fix successful! The dashboard should work now.")
    else:
        print("\n⚠️  API still has issues. Check the server logs.")