# EXPERIMENT 2: Effect of top_k
# =================================================

import time

questions = model_t_questions[:3]  # use 3–5 questions
k_values = [1, 3, 5, 10, 20]

if index.ntotal > 0:
    for q in questions:
        print("\n" + "="*90)
        print("QUESTION:", q)

        for k in k_values:
            print(f"\n{'-'*60}")
            print(f"TOP_K = {k}")

            start = time.time()
            answer = rag_query(q, top_k=k)
            latency = time.time() - start

            print("\nAnswer:")
            print(answer)
            print(f"\nLatency: {latency:.2f} seconds")

        print("="*90)
else:
    print("Please complete the pipeline setup first.")