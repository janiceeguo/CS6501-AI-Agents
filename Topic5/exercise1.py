# # EXPERIMENT 1: Compare WITH vs WITHOUT RAG
# # ==========================================

# Run for Model T
model_t_questions = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]

def compare_no_rag_vs_rag(questions, top_k=5, show_context=True):
    if index.ntotal == 0:
        print("Please complete the pipeline setup first (index is empty).")
        return

    for question in questions:
        print("\n" + "=" * 90)
        print("QUESTION:", question)

        # WITHOUT RAG
        direct_prompt = f"""Answer this question:
{question}

Answer:"""
        print("\nWITHOUT RAG (model's own knowledge):")
        print("-" * 40)
        direct_answer = generate_response(direct_prompt)
        print(direct_answer)

        print("\n" + "-" * 40)

        # WITH RAG
        print("WITH RAG (using retrieved context):")
        print("-" * 40)
        rag_answer = rag_query(question, top_k=top_k, show_context=show_context)
        print(rag_answer)

        print("=" * 90)

compare_no_rag_vs_rag(model_t_questions, top_k=5, show_context=True)


cr_questions = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake Elise Stefanovic make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]

# Run for Congressional Record
compare_no_rag_vs_rag(cr_questions, top_k=5, show_context=True)

