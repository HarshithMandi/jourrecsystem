import requests
import json

base_url = "http://localhost:8000/api"

abstract = "Machine learning models for natural language processing tasks have shown significant improvements in recent years through the use of transformer architectures and pre-trained language models."

print("Testing /api/recommend-detailed endpoint...")
try:
    response = requests.post(
        f"{base_url}/recommend-detailed",
        json={"abstract": abstract, "top_k": 5},
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Got {len(data.get('recommendations', []))} recommendations")
        print(f"Processing time: {data.get('processing_time_ms')}ms")
    else:
        print(f"Error: {response.text}")
except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect to API server. Is it running?")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*50 + "\n")

print("Testing /api/compare-rankings endpoint...")
try:
    response = requests.post(
        f"{base_url}/compare-rankings",
        json={"abstract": abstract, "top_k": 5},
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Got comparisons")
        print(f"Keys: {list(data.get('comparisons', {}).keys())}")
        print(f"Processing time: {data.get('processing_time_ms')}ms")
    else:
        print(f"Error: {response.text}")
except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect to API server. Is it running?")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*50 + "\n")

print("Testing /api/analyze-text endpoint...")
try:
    response = requests.post(
        f"{base_url}/analyze-text",
        json={"abstract": abstract, "top_k": 5},
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Text analysis complete")
        print(f"Keys: {list(data.keys())}")
        print(f"Processing time: {data.get('processing_time_ms')}ms")
    else:
        print(f"Error: {response.text}")
except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect to API server. Is it running?")
except Exception as e:
    print(f"ERROR: {e}")
