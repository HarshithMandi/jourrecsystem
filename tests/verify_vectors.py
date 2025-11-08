"""Quick verification script to check vector dimensions"""
import sqlite3
import json

# Connect to database
conn = sqlite3.connect('data/journal_rec_crossdisciplinary.db')
cursor = conn.cursor()

# Get 10 random journals with their vectors
cursor.execute("""
    SELECT j.name, jp.bert_vector, jp.tfidf_vector
    FROM journals j
    JOIN journal_profiles jp ON j.id = jp.journal_id
    WHERE jp.bert_vector IS NOT NULL 
    AND jp.tfidf_vector IS NOT NULL
    LIMIT 10
""")

print("Checking vector dimensions for 10 sample journals:\n")
all_consistent = True
tfidf_dim = None
bert_dim = None

for name, bert_json, tfidf_json in cursor:
    bert_vec = json.loads(bert_json)
    tfidf_vec = json.loads(tfidf_json)
    
    bert_len = len(bert_vec)
    tfidf_len = len(tfidf_vec)
    
    if bert_dim is None:
        bert_dim = bert_len
    if tfidf_dim is None:
        tfidf_dim = tfidf_len
    
    status = "✓" if (bert_len == 384 and tfidf_len == tfidf_dim) else "✗"
    print(f"{status} {name[:50]:50} BERT: {bert_len}, TF-IDF: {tfidf_len}")
    
    if bert_len != bert_dim or tfidf_len != tfidf_dim:
        all_consistent = False

# Get total counts
cursor.execute("SELECT COUNT(*) FROM journals")
total_journals = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*)
    FROM journals j
    JOIN journal_profiles jp ON j.id = jp.journal_id
    WHERE jp.bert_vector IS NOT NULL 
    AND jp.tfidf_vector IS NOT NULL
""")
vectorized_journals = cursor.fetchone()[0]

print(f"\n{'='*70}")
print(f"Database: data/journal_rec_crossdisciplinary.db")
print(f"Total journals: {total_journals}")
print(f"Journals with vectors: {vectorized_journals}")
print(f"Expected BERT dimensions: 384")
print(f"Expected TF-IDF dimensions: 1651 (based on corpus)")
print(f"Actual BERT dimensions: {bert_dim}")
print(f"Actual TF-IDF dimensions: {tfidf_dim}")
print(f"\nAll vectors consistent: {'YES ✓' if all_consistent else 'NO ✗'}")
print(f"{'='*70}")

conn.close()
