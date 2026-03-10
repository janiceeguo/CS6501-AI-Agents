chunks = [128, 512, 2048]
questions = [
    "How do I adjust the headlamps?",
    "What causes a noisy time gear?",
    "What is the difference between the two methods of replacing the top tank top?",
    "The engine of the car isn't starting, what should I do to narrow down the root cause of this issue?",
    "How often should I check the battery?"
]
for i in chunks:
    if i == 128:
        rebuild_pipeline(chunk_size=i, chunk_overlap=64)
    else:
        rebuild_pipeline(chunk_size=i, chunk_overlap=128)
    for q in questions:
        print(f"Question: {q}\n")
        rag_answer = rag_query(q, top_k=5)
        print(rag_answer)
        print("=" * 60)