from openai import OpenAI
import os

from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

client = OpenAI()

def gpt4o_mini_query(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0  # important for fair comparison
    )
    return response.choices[0].message.content

def run_gpt4o_mini(questions):
    for q in questions:
        print("\n" + "="*90)
        print("QUESTION:", q)
        print("\nGPT-4o Mini (NO RAG)")
        print("-"*40)
        print(gpt4o_mini_query(q))
        print("="*90)

# Model T
run_gpt4o_mini(model_t_questions)

# Congressional Record
run_gpt4o_mini(cr_questions)