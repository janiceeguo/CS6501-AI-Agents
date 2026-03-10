questions = [
    "What are ALL the maintenance tasks I need to do monthly?",
    "Compare the procedures for adjusting the front spring vs. adjusting the rear spring",
    "What tools do I need for a complete tune-up?",
    "Summarize all safety warnings in the manual"
]
ks = [3, 5, 10]
for i in ks:
    print("=" * 60)
    print(f"TOP_K = {i}")
    print("-" * 60)
    for q in questions:
        print(f"Question: {q}\n")
        rag_answer = rag_query(q, top_k=i)
        print(f"Answer: {rag_answer}")
        print("-" * 60)