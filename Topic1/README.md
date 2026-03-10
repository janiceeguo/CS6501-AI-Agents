# Running an LLM

- Jupyter notebook (all code) in `llm.ipynb`
- Separate code and models for each task in each `.py` file
- Separate outputs for timings and chat agent in the `outputs` folder
- All graphs for task 6 in the `graphs` folder

## Task 4:

For the timings, my computer does not have a GPU. Thus, I was only able to run CPU on my local. Quantiztion was generally slower than no quantization, and CPU timings were much slower than GPU timings. My local CPU timings were the slowest, even compared to running CPU only in Colab.

## Task 6:

Across the evaluated MMLU subjects, **Llama 3.2-1B-Instruct** and **Qwen** achieve similar overall accuracy (≈42–43%), substantially outperforming **TinyLlama** (≈24%). The similarity in overall performance between Llama and Qwen suggests that models of similar scale and training quality can reach similar levels of general knowledge performance even if their domain-specific strengths differ, most likely reflecting differences in training data composition rather than fundamental capability differences. Their moderate error correlation (0.32) and the relatively large number of shared mistakes indicate that many difficult questions expose systematic reasoning limitations common to both models, rather than idiosyncratic weaknesses. In other words, these models tend to struggle on the same conceptual gaps (likely questions requiring deeper multi-step reasoning or less common domain knowledge) suggesting that their architectural similarity leads to similar failure modes. However, the presence of unique errors for each model and only partial overlap in incorrect answers implies that each model still captures somewhat different knowledge distributions, meaning that model diversity could improve coverage on challenging questions.

In contrast, **TinyLlama**’s much weaker performance suggests a qualitatively different behavior. Rather than failing on the same difficult reasoning tasks, TinyLlama appears limited primarily by capacity constraints, causing it to miss both easy and difficult questions more indiscriminately. This interpretation is reinforced by the large number of errors unique to TinyLlama (270) and its heavy participation in overlaps involving many questions missed by multiple models. The answer distribution analysis further supports this interpretation: whereas Llama and Qwen distribute incorrect responses relatively evenly across options (meaning that they attempted to reason but reached incorrect conclusions) TinyLlama’s strong skew toward option C indicates a systematic bias. This kind of behavior often emerges when smaller models lack sufficient confidence or representation to differentiate answer choices, causing them to default disproportionately to a particular token pattern rather than evaluating the alternatives meaningfully. Taken together, these patterns suggest that while Llama and Qwen primarily fail due to shared reasoning or knowledge limitations, TinyLlama’s failures are more strongly driven by model capacity and biased guessing behavior, showing how model scale affects not just accuracy but also the structure of mistakes.

## Task 8:

I chose to enable a fixed window context, where history is maintained by a maximum number of conversations kept. Without history, the model is able to reference facts given only in the current query, and is unable to connect to previous questions. With history, even without explicitly prompting the model to recall certain facts, it will reference them in following conversations when appropriate, showing ability to recall information. For example, when asking about a dog named Max, the model continues to refer to the dog as Max, even when I only use pronouns in my follow up question.