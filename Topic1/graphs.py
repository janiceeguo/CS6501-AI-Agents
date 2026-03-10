import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load model results
# -------------------------------

files = {
    "Llama": "/content/llama_colab_gpu_8bit_answers.json",
    "Qwen": "/content/qwen_colab_gpu_8bit_answers.json",
    "Tiny_llama": "/content/tiny_colab_gpu_8bit_answers.json"
}

models = {}

for model_name, path in files.items():
    with open(path) as f:
        data = json.load(f)

    rows = []

    for subject in data["subject_results"]:
        for q in subject["questions"]:
            rows.append({
                "model": model_name,
                "subject": subject["subject"],
                "question_number": q["question_number"],
                "ground_truth": q["ground_truth"],
                "prediction": q["model_answer"],
                "correct": q["correct"]
            })

    models[model_name] = pd.DataFrame(rows)

df = pd.concat(models.values())

print(df.head())

accuracy = df.groupby("model")["correct"].mean() * 100

plt.figure()
accuracy.plot(kind="bar")
plt.ylabel("Accuracy (%)")
plt.title("Overall MMLU Accuracy")
plt.show()

subject_acc = df.groupby(["model","subject"])["correct"].mean().reset_index()

plt.figure(figsize=(8,4))
sns.barplot(data=subject_acc, x="subject", y="correct", hue="model")
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylabel("Accuracy")
plt.title("Accuracy by Subject")
plt.show()

mistakes = {}

for model in files.keys():
    mistakes[model] = set(
        df[(df.model==model) & (df.correct==False)]
        .apply(lambda r: (r.subject, r.question_number), axis=1)
    )

models_list = list(files.keys())

overlap = pd.DataFrame(index=models_list, columns=models_list)

for m1 in models_list:
    for m2 in models_list:
        overlap.loc[m1,m2] = len(mistakes[m1].intersection(mistakes[m2]))

print(overlap)

from matplotlib_venn import venn3

plt.figure()

venn3([
    mistakes["Llama"],
    mistakes["Qwen"],
    mistakes["Tiny_llama"]
],
set_labels=("Llama","Qwen","Tiny_llama"))

plt.title("Shared Mistakes Between Models")
plt.show()

wrong = df[df.correct==False]

letter_dist = wrong.groupby(["model","prediction"]).size().unstack().fillna(0)

letter_dist.plot(kind="bar", stacked=True)

plt.title("Wrong Answer Distribution")
plt.ylabel("Count")
plt.show()

pivot = df.pivot_table(
    index=["subject","question_number"],
    columns="model",
    values="correct"
)

pivot["num_correct"] = pivot.sum(axis=1)

hard_questions = pivot[pivot["num_correct"] == 0]

print("Questions all models got wrong:")
print(hard_questions)

pivot = df.pivot_table(
    index=["subject","question_number"],
    columns="model",
    values="correct"
)

pivot["num_correct"] = pivot.sum(axis=1)

hard_questions = pivot[pivot["num_correct"] == 0]

print("Questions all models got wrong:")
print(hard_questions)

pivot["num_correct"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Number of Models Correct")
plt.ylabel("Questions")
plt.title("Difficulty Distribution")
plt.show()

mistake_matrix = df.copy()
mistake_matrix["wrong"] = ~mistake_matrix["correct"]

pivot = mistake_matrix.pivot_table(
    index=["subject","question_number"],
    columns="model",
    values="wrong"
)

corr = pivot.corr()

sns.heatmap(corr, annot=True, vmin=0, vmax=1)
plt.title("Correlation of Model Mistakes")
plt.show()