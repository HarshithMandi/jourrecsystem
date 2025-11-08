from app.services.recommender import get_ranking_comparisons, rank_journals

abstract = "Machine learning models for natural language processing tasks have shown significant improvements in recent years."

print("Testing get_ranking_comparisons...")
try:
    result = get_ranking_comparisons(abstract, 5)
    print(f"Success! Keys: {list(result.keys())}")
    print(f"Similarity: {len(result['similarity_ranking'])} journals")
    print(f"TF-IDF: {len(result['tfidf_only_ranking'])} journals")
    print(f"BERT: {len(result['bert_only_ranking'])} journals")
    print(f"Impact: {len(result['impact_factor_ranking'])} journals")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")
print("Testing rank_journals...")
try:
    result = rank_journals(abstract, 5)
    print(f"Success! Got {len(result)} journals")
    if result:
        print(f"First journal: {result[0]['journal_name']}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
