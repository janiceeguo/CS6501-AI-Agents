# Agent Orchestration Frameworks:

- Jupyter notebook (all code) in `agent_control.ipynb`
- Separate code and models for each task in each `.py` file
- Separate outputs and graphs for each task in the `outputs` folder

## Task 2

Originally, the program handles empty input as another type of input. It tends to generate random conversations and context for each conversation, and does not take in previous queries. This reveals that less sophisticated LLMs have limited capacity in producing cohesive responses, as they are not able to remember previous history and will simply generate their own responses.