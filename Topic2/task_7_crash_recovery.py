# LangGraph dual-agent chat with:
# - Llama + Qwen
# - tri-party chat illusion
# - SQLite crash recovery (resume mid-conversation)

import torch
from typing import TypedDict, Annotated, Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# =============================================================================
# DEVICE SELECTION
# =============================================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# STATE
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    user_input: str
    active_model: str
    should_exit: bool
    verbose: bool


# =============================================================================
# LLM CREATION
# =============================================================================
def create_llm(model_id: str):
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=pipe)


# =============================================================================
# GRAPH
# =============================================================================
def create_graph(llama_llm, qwen_llm):

    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 60)
        print("Enter text (Hey Llama / Hey Qwen, quit to exit)")
        print("=" * 60)
        print("> ", end="")

        text = input().strip()

        if text.lower() in {"quit", "exit", "q"}:
            return {"should_exit": True}

        active_model = state["active_model"]
        if text.lower().startswith("hey llama"):
            active_model = "Llama"
            text = text[len("hey llama"):].strip()
        elif text.lower().startswith("hey qwen"):
            active_model = "Qwen"
            text = text[len("hey qwen"):].strip()
        else:
            active_model = "Llama"

        return {
            "user_input": text,
            "active_model": active_model,
            "messages": [HumanMessage(content=f"Human: {text}")],
        }

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if not state.get("user_input"):
            return "get_user_input"
        return "call_llm"

    def call_llm(state: AgentState) -> dict:
        model_name = state["active_model"]
        messages = state["messages"]

        system_prompt = (
            f"You are {model_name}. Participants are Human, Llama, and Qwen.\n"
            "Reply with exactly ONE line.\n"
            f"That line MUST start with '{model_name}: '."
            "Do NOT write any other speaker lines (no 'Human:' or 'Llama:').\n"
            "Do NOT continue the conversation beyond your one line."
        )

        prompt_parts = [f"System: {system_prompt}"]
        for msg in messages:
            if msg.content.startswith(model_name + ":"):
                prompt_parts.append(f"Assistant: {msg.content}")
            else:
                prompt_parts.append(f"User: {msg.content}")

        assistant_prompt = f"\nAssistant: {model_name}:"
        prompt = "\n".join(prompt_parts) + assistant_prompt

        llm = llama_llm if model_name == "Llama" else qwen_llm
        full_response = llm.invoke(prompt)
        response = full_response.split(assistant_prompt)[-1].strip()

        return {
            "messages": [HumanMessage(content=f"{model_name}: {response}")]
        }

    def print_response(state: AgentState) -> dict:
        for msg in reversed(state["messages"]):
            if msg.content.startswith(("Llama:", "Qwen:")):
                print("\n" + "=" * 70)
                print(msg.content)
                print("=" * 70)
                break
        return {}

    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_llm", call_llm)
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_llm": "call_llm",
            END: END,
        },
    )
    builder.add_edge("call_llm", "print_response")
    builder.add_edge("print_response", "get_user_input")

    # SQLite checkpointer
    checkpointer = SqliteSaver.from_conn_string("chat_checkpoints.db")
    # return builder.compile(checkpointer=checkpointer)
    return builder


# =============================================================================
# MAIN (CRASH-SAFE)
# =============================================================================
# def main():
#     llama = create_llm("meta-llama/Llama-3.2-1B-Instruct")
#     qwen = create_llm("Qwen/Qwen2.5-0.5B-Instruct")

#     graph = create_graph(llama, qwen)

#     config = {"configurable": {"thread_id": "chat-session-1"}}

#     try:
#         state = graph.get_state(config)

#         if state.next:
#             print("\nResuming conversation from checkpoint...")
#             graph.invoke(None, config=config)
#         else:
#             print("\nStarting new conversation...")
#             graph.invoke(
#                 {
#                     "messages": [],
#                     "user_input": "",
#                     "active_model": "Llama",
#                     "should_exit": False,
#                     "verbose": False,
#                 },
#                 config=config,
#             )

#     except SystemExit as e:
#         print("\nProgram crashed:", e)
#         print("State saved. Restart to resume.")
def main():
    llama = create_llm("meta-llama/Llama-3.2-1B-Instruct")
    qwen = create_llm("Qwen/Qwen2.5-0.5B-Instruct")

    builder = create_graph(llama, qwen)

    # USE CONTEXT MANAGER HERE
    with SqliteSaver.from_conn_string("chat_checkpoints.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "chat-session-1"}}

        try:
            state = graph.get_state(config)

            if state.next:
                print("\nResuming conversation from checkpoint...")
                graph.invoke(None, config=config)
            else:
                print("\nStarting new conversation...")
                graph.invoke(
                    {
                        "messages": [],
                        "user_input": "",
                        "active_model": "Llama",
                        "should_exit": False,
                        "verbose": False,
                    },
                    config=config,
                )

        except SystemExit as e:
            print("\nProgram crashed:", e)
            print("State saved. Restart to resume.")


if __name__ == "__main__":
    main()
