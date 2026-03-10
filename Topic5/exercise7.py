overlaps = [0, 64, 128, 256]
questions = [
    "What are the essentials of good service?",
    "What information about car maintenance or care should be included in the first letter sent out after the delivery of a new car?",
    "What are some things to be aware of before or while performing major repair operations?"
]
for i in overlaps:
    rebuild_pipeline(chunk_size=512, chunk_overlap=i)
    for q in questions:
        print(f"Question: {q}\n")
        rag_answer = rag_query(q, top_k=5)
        print(rag_answer)
        print("=" * 60)