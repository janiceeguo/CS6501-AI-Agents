unanswerable_questions = [
    # Completely off-topic
    "What is the capital of France?",

    # Related but likely not in the manual (depends on your manual content)
    "What's the horsepower of a 1925 Model T?",

    # False premise / leading
    "Why does the manual recommend synthetic oil?",
    "Which section says to use 5W-30 full synthetic oil?",
]

def test_unanswerables(questions, top_k=5, show_context=True):
    if index.ntotal == 0:
        print("Index empty — build the RAG pipeline first.")
        return

    for q in questions:
        print("\n" + "="*90)
        print("QUESTION:", q)

        # NO RAG
        print("\nNO RAG:")
        print("-"*40)
        direct_prompt = f"""Answer this question:
{q}

Answer:"""
        print(generate_response(direct_prompt))

        # WITH RAG
        print("\nWITH RAG:")
        print("-"*40)
        print(rag_query(q, top_k=top_k, show_context=show_context))

test_unanswerables(unanswerable_questions, top_k=5, show_context=True)