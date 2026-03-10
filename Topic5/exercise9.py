questions = [
    "How do I adjust the headlamps?",
    "What causes a noisy time gear?",
    "What is the difference between the two methods of replacing the top tank top?",
    "The engine of the car isn't starting, what should I do to narrow down the root cause of this issue?",
    "How often should I check the battery?",
    "How do I adjust the carburetor?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in the engine?",
    "What is the steering gear ratio of the newest car design?"
]

import numpy as np
from collections import Counter

for q in questions:
    if index.ntotal > 0:
        results = retrieve(q, top_k=10)

        print(f"Query: {q}\n")
        print("Top 10 retrieved chunks:")

        scores = []

        for i, (chunk, score) in enumerate(results, 1):
            scores.append(score)

            print(f"\n[{i}] Score: {score:.4f} | Source: {chunk.source_file}")

        # ----- Score Distribution -----
        if scores:
            scores_array = np.array(scores)

            print("\nScore Distribution:")
            print(f"  Min:  {scores_array.min():.4f}")
            print(f"  Max:  {scores_array.max():.4f}")
            print(f"  Mean: {scores_array.mean():.4f}")
            print(f"  Std:  {scores_array.std():.4f}")

            # Simple bucketed histogram (rounded to 2 decimals)
            rounded_scores = [round(s, 2) for s in scores]
            distribution = Counter(rounded_scores)

            print("\n  Rounded Score Frequency:")
            for score_value, count in sorted(distribution.items()):
                print(f"    {score_value:.2f}: {count}")

            # Optional: show percentiles
            print("\n  Percentiles:")
            for p in [25, 50, 75, 90]:
                print(f"    {p}th: {np.percentile(scores_array, p):.4f}")

        rag_answer = rag_query(q, top_k=10)
        print(f"\nAnswer: {rag_answer}")

        print("=" * 60)

    else:
        print("Index is empty - please load, chunk, and embed documents first.")



# EXPERIMENT

questions = [
    "How do I adjust the headlamps?",
    "What causes a noisy time gear?",
    "What is the difference between the two methods of replacing the top tank top?",
    "The engine of the car isn't starting, what should I do to narrow down the root cause of this issue?",
    "How often should I check the battery?",
    "How do I adjust the carburetor?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in the engine?",
    "What is the steering gear ratio of the newest car design?"
]

import numpy as np
from collections import Counter

THRESHOLD = 0.5

for q in questions:
    if index.ntotal > 0:
        results = retrieve(q, top_k=10)

        print(f"\nQuery: {q}\n")

        # Filter results by threshold
        filtered_results = [
            (chunk, score) for chunk, score in results
            if score > THRESHOLD
        ]

        if not filtered_results:
            print("No chunks passed the threshold.")
            direct_answer = direct_query(q)
            print(f"\nAnswer: {direct_answer}")
            continue

        print(f"Chunks with score > {THRESHOLD}:")

        for i, (chunk, score) in enumerate(filtered_results, 1):
            print(f"\n[{i}] Score: {score:.4f} | Source: {chunk.source_file}")

        # Optional: show how many were filtered out
        print(f"\nKept {len(filtered_results)} of {len(results)} chunks.")

         # ----- Score Distribution -----
        if scores:
            scores_array = np.array(scores)

            print("\nScore Distribution:")
            print(f"  Min:  {scores_array.min():.4f}")
            print(f"  Max:  {scores_array.max():.4f}")
            print(f"  Mean: {scores_array.mean():.4f}")
            print(f"  Std:  {scores_array.std():.4f}")

            # Simple bucketed histogram (rounded to 2 decimals)
            rounded_scores = [round(s, 2) for s in scores]
            distribution = Counter(rounded_scores)

            print("\n  Rounded Score Frequency:")
            for score_value, count in sorted(distribution.items()):
                print(f"    {score_value:.2f}: {count}")

            # Optional: show percentiles
            print("\n  Percentiles:")
            for p in [25, 50, 75, 90]:
                print(f"    {p}th: {np.percentile(scores_array, p):.4f}")

        rag_answer = rag_query(q, top_k=len(filtered_results))
        print(f"\nAnswer: {rag_answer}")

        print("=" * 60)

    else:
        print("Index is empty - please load, chunk, and embed documents first.")