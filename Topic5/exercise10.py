template_minimal = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
template_strict = """You are a helpful assistant.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
Answer ONLY based on the context above.
If the answer is not in the context, say:
"The context does not contain enough information to answer this question."

ANSWER:"""
template_citation = """You are a research assistant.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
Answer the question using only the context.
Quote the exact passages from the context that support your answer.
If the answer is not in the context, say so explicitly.

ANSWER (with citations):"""
template_permissive = """You are a knowledgable assistant.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
Use the provided context to help answer the question.
You may also use your general knowledge if needed.
Clearly distinguish between information from the context and your own knowledge.

ANSWER:"""
template_structured = """You are an analytical assistant.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
You MUST follow the exact output format below. Do NOT skip sections.
First, list the relevant facts from the context.
Then, synthesize your answer to the question based only on those facts.

OUTPUT FORMAT:
  RELEVANT FACTS:
  FINAL ANSWER:"""
templates = {
    "MINIMAL": template_minimal,
    "STRICT GROUNDING": template_strict,
    "ENCOURAGING CITATION": template_citation,
    "PERMISSIVE": template_permissive,
    "STRUCTURED OUTPUT": template_structured
}
questions = [
    "How do I adjust the headlamps?",
    "What causes a noisy time gear?",
    "How do I fix a slipping transmission band?",
    "The engine of the car isn't starting, what should I do to narrow down the root cause of this issue?",
    "How often should I check the battery?"
]
for k, v in templates.items():
    print("=" * 60)
    print(f"TEMPLATE: {k}")
    print("-" * 60)
    print(v)
    print("=" * 60)
    for q in questions:
        print(f"Question: {q}\n")
        rag_answer = rag_query(q, top_k=5, prompt_template=v)
        print(f"Answer: {rag_answer}")
        print("-" * 60)