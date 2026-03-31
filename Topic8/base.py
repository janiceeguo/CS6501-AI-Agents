import random
import tinker
from tinker import types
import numpy as np
import json, random
from sql_matches import sql_matches

with open("/content/sql_create_context_4v.json") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"\nSample example:")
ex = data[0]
print(f"  Question: {ex['question']}")
print(f"  Context:  {ex['context'][:120]}...")
print(f"  Answer:   {ex['answer']}")

NUM_TEST_EXAMPLES = 200  # Held-out for evaluation; all remaining data used for training
random.shuffle(data)
test_data = data[:NUM_TEST_EXAMPLES]
train_data = data[NUM_TEST_EXAMPLES:]
print(f"Training examples: {len(train_data)} (all except evaluation)")
print(f"Test examples: {len(test_data)}")

def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    """Generate SQL from the model given schema and question."""
    prompt = f"""Table schema:
{context}
Question: {question}
SQL: """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    params = types.SamplingParams(max_tokens=150, temperature=0.0, stop=["\n\n", "Question:"])
    result = sampling_client.sample(prompt=model_input, sampling_params=params, num_samples=1).result()
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""

def eval_one(sampling_client, tokenizer, ex: dict) -> bool:
    """Evaluate one example: generate SQL, then check if it matches expected."""
    sql = sample_from_model(sampling_client, tokenizer, ex["context"], ex["question"])
    return sql_matches(sql, ex["answer"], schema=ex["context"])

def evaluate_test_set(sampling_client, tokenizer, test_data: list) -> float:
    """Compute accuracy = fraction of test examples where generated SQL matches expected."""
    correct = sum(1 for ex in test_data if eval_one(sampling_client, tokenizer, ex))
    return correct / len(test_data)



service_client = tinker.ServiceClient()
base_model = "meta-llama/Llama-3.2-1B"
training_client = service_client.create_lora_training_client(base_model=base_model)
tokenizer = training_client.get_tokenizer()

print("\n--- Evaluating Base Model on 200 Test Questions ---")
base_sampling_client = training_client.save_weights_and_get_sampling_client(
    name="base-model"
)
base_accuracy = evaluate_test_set(
    base_sampling_client, tokenizer, test_data
)
print(f"Base model accuracy: {base_accuracy:.2%} ({int(base_accuracy * 200)}/200)")