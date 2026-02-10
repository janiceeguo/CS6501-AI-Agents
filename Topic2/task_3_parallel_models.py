# langgraph_simple_agent.py
# LangGraph simple agent with:
# - verbose / quiet tracing
# - empty input handled via graph self-loop
# - parallel execution of Llama + Qwen models

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# =============================================================================
# DEVICE SELECTION
# =============================================================================
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"


# =============================================================================
# STATE DEFINITION
# =============================================================================
class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    verbose: bool
    llama_response: str
    qwen_response: str


# =============================================================================
# LLM CREATION
# =============================================================================
def create_llm(model_id: str):
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=pipe)


# =============================================================================
# GRAPH CREATION
# =============================================================================
def create_graph(llama_llm, qwen_llm):

    # -------------------------------------------------------------------------
    # NODE: get_user_input
    # -------------------------------------------------------------------------
    def get_user_input(state: AgentState) -> dict:
        if state["verbose"]:
            print("[TRACE] Node: get_user_input")

        print("\n" + "=" * 50)
        print("Enter text (quit/exit/q to leave)")
        print("Type 'verbose' or 'quiet' to toggle tracing")
        print("=" * 50)
        print("> ", end="")

        user_input = input().strip()

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return {"should_exit": True}

        if user_input.lower() == "verbose":
            print("Verbose tracing ENABLED")
            return {"verbose": True, "user_input": ""}

        if user_input.lower() == "quiet":
            print("Verbose tracing DISABLED")
            return {"verbose": False, "user_input": ""}

        return {
            "user_input": user_input,
            "should_exit": False,
        }

    # -------------------------------------------------------------------------
    # ROUTER: 3-WAY CONDITIONAL
    # -------------------------------------------------------------------------
    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END

        if state.get("user_input", "") == "":
            return "get_user_input"

        return "dispatch_models"

    # -------------------------------------------------------------------------
    # NODE: dispatch_models (fan-out point)
    # -------------------------------------------------------------------------
    def dispatch_models(state: AgentState) -> dict:
        if state["verbose"]:
            print("[TRACE] Dispatching input to Llama and Qwen")
        return {}

    # -------------------------------------------------------------------------
    # NODE: call_llama
    # -------------------------------------------------------------------------
    def call_llama(state: AgentState) -> dict:
        if state["verbose"]:
            print("[TRACE] Node: call_llama")

        prompt = f"User: {state['user_input']}\nAssistant:"
        response = llama_llm.invoke(prompt)
        return {"llama_response": response}

    # -------------------------------------------------------------------------
    # NODE: call_qwen
    # -------------------------------------------------------------------------
    def call_qwen(state: AgentState) -> dict:
        if state["verbose"]:
            print("[TRACE] Node: call_qwen")

        prompt = f"User: {state['user_input']}\nAssistant:"
        response = qwen_llm.invoke(prompt)
        return {"qwen_response": response}

    # -------------------------------------------------------------------------
    # NODE: print_both_responses (fan-in)
    # -------------------------------------------------------------------------
    def print_both_responses(state: AgentState) -> dict:
        if state["verbose"]:
            print("[TRACE] Node: print_both_responses")

        print("\n" + "=" * 50)
        print("LLAMA RESPONSE")
        print("=" * 50)
        print(state.get("llama_response", ""))

        print("\n" + "=" * 50)
        print("QWEN RESPONSE")
        print("=" * 50)
        print(state.get("qwen_response", ""))

        return {}

    # -------------------------------------------------------------------------
    # GRAPH CONSTRUCTION
    # -------------------------------------------------------------------------
    graph = StateGraph(AgentState)

    graph.add_node("get_user_input", get_user_input)
    graph.add_node("dispatch_models", dispatch_models)
    graph.add_node("call_llama", call_llama)
    graph.add_node("call_qwen", call_qwen)
    graph.add_node("print_both_responses", print_both_responses)

    graph.add_edge(START, "get_user_input")

    graph.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "dispatch_models": "dispatch_models",
            END: END,
        },
    )

    # Fan-out
    graph.add_edge("dispatch_models", "call_llama")
    graph.add_edge("dispatch_models", "call_qwen")

    # Fan-in
    graph.add_edge("call_llama", "print_both_responses")
    graph.add_edge("call_qwen", "print_both_responses")

    # Loop
    graph.add_edge("print_both_responses", "get_user_input")

    return graph.compile()


def save_graph_image(graph, filename="lg_graph_3.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need: pip install grandalf")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 50)
    print("LangGraph Parallel LLM Agent")
    print("=" * 50)

    print("\nLoading Llama...")
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct")

    print("\nLoading Qwen...")
    qwen_llm = create_llm("Qwen/Qwen2.5-0.5B-Instruct")

    graph = create_graph(llama_llm, qwen_llm)

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "verbose": False,
        "llama_response": "",
        "qwen_response": "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
