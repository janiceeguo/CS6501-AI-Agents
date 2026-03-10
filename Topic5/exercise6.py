phrasings = {
    "Formal": "What is the correct spark plug gap specification for a Model T Ford?",
    "Casual": "How far apart should the Model T spark plug be?",
    "Keywords": "Model T spark plug gap",
    "Question form": "What gap do I set the spark plugs to?",
    "Indirect": "Ignition system spark plug spacing requirement",
    "More specific": "Spark plug gap in thousandths of an inch for a Model T",
}

def show_top_chunks(query, top_k=5, max_chars=250):
    results = retrieve(query, top_k)
    print("\n" + "="*90)
    print("QUERY:", query)
    print("="*90)
    for i, (chunk, score) in enumerate(results, 1):
        preview = chunk.text.replace("\n", " ").strip()
        preview = preview[:max_chars] + ("..." if len(preview) > max_chars else "")
        print(f"\n#{i}  score={score:.3f}  source={chunk.source_file}")
        print(preview)

all_results = {}  # store for overlap comparisons

for label, q in phrasings.items():
    results = retrieve(q, top_k=5)
    all_results[label] = results
    show_top_chunks(q, top_k=5, max_chars=250)


def chunk_id(chunk):
    # create a simple stable-ish identifier
    return (chunk.source_file, chunk.text[:80])

# Build sets of chunk IDs for each phrasing
result_sets = {
    label: {chunk_id(chunk) for (chunk, score) in results}
    for label, results in all_results.items()
}

labels = list(result_sets.keys())

print("\n" + "="*90)
print("OVERLAP (count of shared chunks in top-5)")
print("="*90)

# Pairwise overlap counts
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        a, b = labels[i], labels[j]
        overlap = len(result_sets[a] & result_sets[b])
        print(f"{a:>12} vs {b:<12}  overlap={overlap}/5")



import numpy as np

print("\n" + "="*90)
print("SCORE SUMMARY")
print("="*90)

for label, results in all_results.items():
    scores = [score for (chunk, score) in results]
    print(f"{label:>12}: max={max(scores):.3f}  mean={np.mean(scores):.3f}  min={min(scores):.3f}")