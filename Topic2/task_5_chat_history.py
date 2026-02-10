# langgraph_simple_agent.py
# LangGraph agent with:
# - persistent chat history via Message API
# - empty-input handled by graph routing
# - single Llama model (Qwen disabled)
# - graph visualization preserved

import torch
from typing import TypedDict, List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

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
# STATE DEFINITION (Message API–based)
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    user_input: str
    should_exit: bool
    verbose: bool


# =============================================================================
# LLM CREATION
# =============================================================================
def create_llm():
    device = get_device()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

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

        text = input().strip()

        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return {"should_exit": True}

        if text.lower() == "verbose":
            print("Verbose tracing ENABLED")
            return {"verbose": True, "user_input": ""}

        if text.lower() == "quiet":
            print("Verbose tracing DISABLED")
            return {"verbose": False, "user_input": ""}

        return {
            "user_input": text,
            "should_exit": False,
            "messages": [HumanMessage(content=text)],
        }

    # -------------------------------------------------------------------------
    # ROUTER: after input (3-way)
    # -------------------------------------------------------------------------
    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END

        if state.get("user_input", "") == "":
            return "get_user_input"

        return "call_llm"

    # -------------------------------------------------------------------------
    # NODE: call_llm (Message API)
    # -------------------------------------------------------------------------
    def call_llm(state: AgentState) -> dict:
        messages = state["messages"]
        verbose = state["verbose"]
        if state["verbose"]:
            print("[TRACE] Node: call_llm")
            print("[TRACE] Messages so far:", len(state["messages"]))

        # Append human message
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        full_response = llm.invoke(prompt)
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            assistant_marker = "\nAssistant:"
            if assistant_marker in full_response:
                parts = full_response.split(assistant_marker)
                response = parts[-1].strip()
            else:
                response = full_response.strip()

        if verbose:
            print(f"[TRACE] LLM response received (length: {len(response)} chars)")
            print("[TRACE] Adding AIMessage to conversation history")
            print("[TRACE] Exiting node: call_llm")

        return {
            "messages": [AIMessage(content=response)]
        }

        # messages = list(state["messages"])
        # messages.append(HumanMessage(content=state["user_input"]))

        # # Invoke LLM with full history
        # ai_text = llm.invoke(messages)

        # # Append AI response
        # messages.append(AIMessage(content=ai_text))

        # return {"messages": messages}

    # -------------------------------------------------------------------------
    # NODE: print_response
    # -------------------------------------------------------------------------
    def print_response(state: AgentState) -> dict:
        last_msg = state["messages"][-1]

        print("\n" + "-" * 50)
        print("Llama:")
        print("-" * 50)
        print(last_msg.content)

        return {}

    # -------------------------------------------------------------------------
    # GRAPH CONSTRUCTION
    # -------------------------------------------------------------------------
    graph = StateGraph(AgentState)

    graph.add_node("get_user_input", get_user_input)
    graph.add_node("call_llm", call_llm)
    graph.add_node("print_response", print_response)

    graph.add_edge(START, "get_user_input")

    graph.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_llm": "call_llm",
            END: END,
        },
    )

    graph.add_edge("call_llm", "print_response")
    graph.add_edge("print_response", "get_user_input")

    return graph.compile()


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
    print("LangGraph Chat Agent (Message API)")
    print("=" * 50)

    llm = create_llm()
    graph = create_graph(llm)
    save_graph_image(graph)

    initial_state: AgentState = {
        "messages": [
            SystemMessage(
                content="You are a helpful, concise assistant."
            )
        ],
        "user_input": "",
        "should_exit": False,
        "verbose": False,
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
