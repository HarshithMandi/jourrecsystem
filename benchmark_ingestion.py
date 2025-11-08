import time
import subprocess

print("Benchmarking ingestion script performance...")
start = time.time()

result = subprocess.run(["python", "scripts/ingest_openalex.py"], 
                       capture_output=False, text=True)

elapsed = time.time() - start
print(f"\n{'='*60}")
print(f"Total execution time: {elapsed:.2f} seconds")
print(f"Performance: ~{353/elapsed:.1f} journals/second processed")
print(f"{'='*60}")
