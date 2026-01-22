# langgraph_dual_agent.py
# LangGraph agent with:
# - persistent tri-party chat history (Human, Llama, Qwen)
# - dynamic switching between Llama and Qwen
# - history rewritten per-target model using Message API
# - per-model system prompts
# - graph visualization preserved

import torch
from typing import TypedDict, List, Tuple, Annotated, Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# =============================================================================
# DEVICE SELECTION
# =============================================================================
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return "mps"
    else:
        print("Using CPU")
        return "cpu"


# =============================================================================
# STATE DEFINITION
# =============================================================================
class AgentState(TypedDict):
    # Canonical chat history: (speaker, text)
    # history: List[Tuple[str, str]]

    # Message API (used only transiently)
    messages: Annotated[Sequence[AnyMessage], add_messages]

    user_input: str
    active_model: str  # "Llama" or "Qwen"
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
# SYSTEM PROMPTS
# =============================================================================
def system_prompt_for(model_name: str) -> str:
    if model_name == "Llama":
        return (
            "You are Llama.\n"
            "Participants in this conversation:\n"
            "- Human (the user)\n"
            "- Llama (you)\n"
            "- Qwen (another AI model)\n\n"
            "All prior messages are prefixed with the speaker name.\n"
            "Respond ONLY as Llama."
        )
    else:
        return (
            "You are Qwen.\n"
            "Participants in this conversation:\n"
            "- Human (the user)\n"
            "- Llama (another AI model)\n"
            "- Qwen (you)\n\n"
            "All prior messages are prefixed with the speaker name.\n"
            "Respond ONLY as Qwen."
        )


# =============================================================================
# HISTORY REWRITE (CRITICAL LOGIC)
# =============================================================================
def build_messages_for_model(
    history: List[Tuple[str, str]],
    target_model: str,
) -> List[AnyMessage]:
    """
    Convert canonical (speaker, text) history into Message API format
    for the target model.
    """

    messages: List[AnyMessage] = [
        SystemMessage(content=system_prompt_for(target_model))
    ]

    for speaker, text in history:
        prefixed = f"{speaker}: {text}"

        if speaker == target_model:
            messages.append(AIMessage(content=prefixed))
        else:
            messages.append(HumanMessage(content=prefixed))

    return messages


# =============================================================================
# GRAPH CREATION
# =============================================================================
def create_graph(llama_llm, qwen_llm):

    # -------------------------------------------------------------------------
    # NODE: get_user_input
    # -------------------------------------------------------------------------
    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 60)
        print("Enter text (Hey Llama / Hey Qwen to switch, quit to exit)")
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

        # history = list(state["history"])
        # if text:
        #     history.append(("Human", text))

        return {
            "user_input": text,
            "active_model": active_model,
            # "history": history,
            "messages": [HumanMessage(content=f"Human: {text}")],
        }

    # -------------------------------------------------------------------------
    # ROUTER
    # -------------------------------------------------------------------------
    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if not state.get("user_input"):
            return "get_user_input"
        return "call_llm"

    # -------------------------------------------------------------------------
    # NODE: call_llm
    # -------------------------------------------------------------------------
    def call_llm(state: AgentState) -> dict:
        model_name = state["active_model"]
        # history = state["history"]
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        user_input = state.get("user_input", "")

        # messages = build_messages_for_model(history, model_name)

        llm = llama_llm if model_name == "Llama" else qwen_llm
        if verbose:
            print("\n[TRACE] Entering node: call_llm")
            print(f"[TRACE] Processing {len(messages)} messages for {model_name}")

        if model_name == "Llama":
            system_prompt = (
                "You are Llama. Participants are Human, Llama, and Qwen. "
                "The conversation so far is shown below with prefixes 'Human:', 'Llama:', 'Qwen:'. "
                "You MUST reply with exactly ONE line. "
                "That line MUST start with 'Llama: ' followed by your answer. "
                "Do NOT write any other speaker lines (no 'Human:' or 'Qwen:'). "
                "Do NOT continue the conversation beyond your one line."
            )
        else:
            system_prompt = (
                "You are Qwen. Participants are Human, Llama, and Qwen. "
                "The conversation so far is shown below with prefixes 'Human:', 'Llama:', 'Qwen:'. "
                "You MUST reply with exactly ONE line. "
                "That line MUST start with 'Qwen: ' followed by your answer. "
                "Do NOT write any other speaker lines (no 'Human:' or 'Llama:'). "
                "Do NOT continue the conversation beyond your one line."
            )

        # response = llm.invoke(messages)

        # history = list(history)
        # history.append((model_name, response))
        prompt_parts = [f"System: {system_prompt}"]

        for msg in messages:
            content = msg.content
            if content.startswith("Human:"):
                prompt_parts.append(f"User: {content}")
            elif model_name == "Llama":
                if content.startswith("Llama:"):
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(f"User: {content}")
            elif model_name == "Qwen":
                if content.startswith("Qwen:"):
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(f"User: {content}")

        if model_name == "Llama":
            assistant_prompt = "\nAssistant: Llama:"
        else:
            assistant_prompt = "\nAssistant: Qwen:"
        prompt = "\n".join(prompt_parts) + assistant_prompt

        full_response = llama_llm.invoke(prompt)

        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            parts = full_response.split(assistant_prompt)
            response = parts[-1].strip() if len(parts) > 1 else full_response.strip()

        if verbose:
            print(f"[TRACE] LLM response: '{response[:100]}...'")
            print(f"[TRACE] Adding HumanMessage with '{model_name}:' prefix")

        return {
            "messages": [HumanMessage(content=f"{model_name}: {response}")]
        }

    # -------------------------------------------------------------------------
    # NODE: print_response
    # -------------------------------------------------------------------------
    def print_response(state: AgentState) -> dict:
        """Prints the most recent AI response."""
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])

        # Find the most recent message that's not from Human
        last_response = None
        for msg in reversed(messages):
            content = msg.content
            if content.startswith("Llama:") or content.startswith("Qwen:"):
                last_response = content
                break

        if verbose:
            print("\n[TRACE] Entering node: print_response")
            print(f"[TRACE] Total messages in history: {len(messages)}")

        print("\n" + "=" * 70)
        print("RESPONSE:")
        print("=" * 70)
        if last_response:
            print(last_response)
        else:
            print("(No response found)")

        if verbose:
            print("\n[TRACE] Response printed to stdout")
            print("[TRACE] Looping back to get_user_input")

        return {}
        # speaker, text = state["history"][-1]

        # print("\n" + "-" * 60)
        # print(f"{speaker}:")
        # print("-" * 60)
        # print(text)

        # return {}

    # -------------------------------------------------------------------------
    # GRAPH
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
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("LangGraph Dual-Agent Chat (Llama + Qwen)")
    print("=" * 60)

    llama = create_llm("meta-llama/Llama-3.2-1B-Instruct")
    qwen = create_llm("Qwen/Qwen2.5-0.5B-Instruct")

    graph = create_graph(llama, qwen)

    initial_state: AgentState = {
        "history": [],
        "messages": [],
        "user_input": "",
        "active_model": "Llama",
        "should_exit": False,
        "verbose": False,
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
