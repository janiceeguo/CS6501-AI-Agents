# langgraph_simple_agent.py
# LangGraph simple agent with:
# - verbose / quiet tracing toggle
# - graph-level handling of empty input via conditional self-loop

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
    llm_response: str
    verbose: bool


# =============================================================================
# LLM CREATION
# =============================================================================
def create_llm():
    device = get_device()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading model: {model_id}")

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
def create_graph(llm):

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

        # Exit commands
        if user_input.lower() in {"quit", "exit", "q"}:
            if state["verbose"]:
                print("[TRACE] Exit requested")
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
            }

        # Verbosity control
        if user_input.lower() == "verbose":
            print("Verbose tracing ENABLED")
            return {
                "user_input": "",
                "should_exit": False,
                "verbose": True,
            }

        if user_input.lower() == "quiet":
            print("Verbose tracing DISABLED")
            return {
                "user_input": "",
                "should_exit": False,
                "verbose": False,
            }

        # Normal input (possibly empty)
        return {
            "user_input": user_input,
            "should_exit": False,
        }

    # -------------------------------------------------------------------------
    # NODE: call_llm
    # -------------------------------------------------------------------------
    def call_llm(state: AgentState) -> dict:
        if state["verbose"]:
            print("[TRACE] Node: call_llm")
            print(f"[TRACE] Input to LLM: {state['user_input']}")

        prompt = f"User: {state['user_input']}\nAssistant:"
        response = llm.invoke(prompt)

        return {"llm_response": response}

    # -------------------------------------------------------------------------
    # NODE: print_response
    # -------------------------------------------------------------------------
    def print_response(state: AgentState) -> dict:
        if state["verbose"]:
            print("[TRACE] Node: print_response")

        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(state["llm_response"])

        return {}

    # -------------------------------------------------------------------------
    # ROUTER: 3-WAY CONDITIONAL
    # -------------------------------------------------------------------------
    def route_after_input(state: AgentState) -> str:
        """
        Routing logic:
        1. should_exit == True        -> END
        2. empty user_input == ""    -> get_user_input (self-loop)
        3. otherwise                 -> call_llm
        """
        if state["verbose"]:
            print("[TRACE] Routing decision")

        if state.get("should_exit", False):
            if state["verbose"]:
                print("[TRACE] Routing to END")
            return END

        if state.get("user_input", "") == "":
            if state["verbose"]:
                print("[TRACE] Empty input detected -> looping to get_user_input")
            return "get_user_input"

        if state["verbose"]:
            print("[TRACE] Routing to call_llm")

        return "call_llm"

    # -------------------------------------------------------------------------
    # GRAPH CONSTRUCTION
    # -------------------------------------------------------------------------
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",  # self-loop
            "call_llm": "call_llm",
            END: END,
        },
    )

    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


# =============================================================================
# GRAPH VISUALIZATION
# =============================================================================
def save_graph_image(graph, filename="lg_graph.png"):
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
    print("LangGraph Simple Agent")
    print("=" * 50)

    llm = create_llm()

    graph = create_graph(llm)
    save_graph_image(graph)

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llm_response": "",
        "verbose": False,
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
