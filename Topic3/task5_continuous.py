"""
Persistent Conversation Agent with LangGraph
- SqliteSaver used correctly as a context manager (with-block)
- Checkpointing and recovery via thread_id
"""

import math
import json
import uuid

from pydantic import BaseModel, Field
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict


# ============================================
# PART 1: Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    result = weather_data.get(location, f"Weather data not available for {location}")
    print(f"  [Tool] get_weather({location!r}) -> {result}")
    return result


class CalculatorInput(BaseModel):
    operation: str = Field(description="Operation to perform: add, subtract, multiply, divide, round, sin, cos, tan, circle_area, rectangle_area, triangle_area, distance_2d")
    value: Optional[float] = Field(default=None, description="Input value for sin, cos, tan, add, subtract, multiply, divide")
    value2: Optional[float] = Field(default=None, description="Second value for add, subtract, multiply, divide")
    decimal_places: Optional[int] = Field(default=0, description="Decimal places for round operation")
    radius: Optional[float] = Field(default=None, description="Radius for circle_area")
    length: Optional[float] = Field(default=None, description="Length for rectangle_area")
    width: Optional[float] = Field(default=None, description="Width for rectangle_area")
    base: Optional[float] = Field(default=None, description="Base for triangle_area")
    height: Optional[float] = Field(default=None, description="Height for triangle_area")
    x1: Optional[float] = Field(default=None, description="X coordinate of point 1 for distance_2d")
    y1: Optional[float] = Field(default=None, description="Y coordinate of point 1 for distance_2d")
    x2: Optional[float] = Field(default=None, description="X coordinate of point 2 for distance_2d")
    y2: Optional[float] = Field(default=None, description="Y coordinate of point 2 for distance_2d")

@tool("calculator", args_schema=CalculatorInput)
def calculator(operation: str, value: float = None, value2: float = None,
               decimal_places: int = None, radius: float = None,
               length: float = None, width: float = None,
               base: float = None, height: float = None,
               x1: float = None, y1: float = None,
               x2: float = None, y2: float = None) -> float:
    """Perform calculator operations. Use the operation name and provide only the required arguments for that operation."""
    if operation == "sin":
        result = math.sin(value)
    elif operation == "cos":
        result = math.cos(value)
    elif operation == "tan":
        result = math.tan(value)
    elif operation == "add":
        result = value + value2
    elif operation == "subtract":
        result = value - value2
    elif operation == "multiply":
        result = value * value2
    elif operation == "divide":
        result = value / value2
    elif operation == "round":
        result = round(value, decimal_places)
    elif operation == "circle_area":
        result = math.pi * radius ** 2
    elif operation == "rectangle_area":
        result = length * width
    elif operation == "triangle_area":
        result = 0.5 * base * height
    elif operation == "distance_2d":
        result = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    print(f"  [Tool] calculator({operation!r}) -> {result}")
    return result


@tool
def count_letter(text: str, letter: str) -> int:
    """Count occurrences of a letter in a text."""
    result = text.lower().count(letter.lower())
    print(f"  [Tool] count_letter({text!r}, {letter!r}) -> {result}")
    return result


@tool
def text_stats(text: str) -> dict:
    """Return word count and character count for a text."""
    words = text.split()
    result = {"words": len(words), "characters": len(text)}
    print(f"  [Tool] text_stats({text!r}) -> {result}")
    return result


TOOLS = [get_weather, calculator, count_letter, text_stats]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant that remembers this entire conversation. "
    "For ANY letter counting use count_letter. "
    "For word/character counts use text_stats. "
    "For ANY math use calculator with the operation name and required arguments. "
    "For weather use get_weather. "
    "Never calculate anything yourself -- always delegate to the tools."
))


# ============================================
# PART 2: State
# ============================================

class AgentState(TypedDict):
    user_input:   str
    should_exit:  bool
    chat_history: list
    llm_response: object


# ============================================
# PART 3: Build Graph
# ============================================

def build_graph(llm):
    llm_with_tools = llm.bind_tools(TOOLS)

    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("You (or 'quit' to exit): ", end="")
        user_input = input().strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True}

        if user_input == "" or user_input.lower() in ("verbose", "quiet"):
            return {"user_input": user_input, "should_exit": False}

        chat_history = state.get("chat_history") or [SYSTEM_PROMPT]
        chat_history = chat_history + [HumanMessage(content=user_input)]

        return {
            "user_input":   user_input,
            "should_exit":  False,
            "chat_history": chat_history,
        }

    def call_llm(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["chat_history"])
        return {
            "llm_response": response,
            "chat_history": state["chat_history"] + [response],
        }

    def call_tools(state: AgentState) -> dict:
        last_msg = state["llm_response"]
        tool_messages = []

        for tc in last_msg.tool_calls:
            name, args = tc["name"], tc["args"]
            result = TOOL_MAP[name].invoke(args) if name in TOOL_MAP \
                     else f"Error: unknown tool {name}"
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        return {"chat_history": state["chat_history"] + tool_messages}

    def print_response(state: AgentState) -> dict:
        print(f"\nAssistant: {state['llm_response'].content}")
        return {}

    def route_user_input(state: AgentState) -> str:
        if state.get("should_exit"):
            return END
        inp = state.get("user_input", "")
        if inp == "" or inp.lower() in ("verbose", "quiet"):
            return "get_user_input"
        return "call_llm"

    def route_after_llm(state: AgentState) -> str:
        if state["llm_response"].tool_calls:
            return "call_tools"
        return "print_response"

    gb = StateGraph(AgentState)

    gb.add_node("get_user_input", get_user_input)
    gb.add_node("call_llm",       call_llm)
    gb.add_node("call_tools",     call_tools)
    gb.add_node("print_response", print_response)

    gb.add_edge(START,            "get_user_input")
    gb.add_edge("call_tools",     "call_llm")
    gb.add_edge("print_response", "get_user_input")

    gb.add_conditional_edges(
        "get_user_input",
        route_user_input,
        {"get_user_input": "get_user_input", "call_llm": "call_llm", END: END},
    )
    gb.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {"call_tools": "call_tools", "print_response": "print_response"},
    )

    return gb


# ============================================
# PART 4: Graph image
# ============================================

def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as exc:
        print(f"Could not save graph image: {exc}")


# ============================================
# PART 5: Main
# ============================================

def main():
    print("=" * 50)
    print("Persistent Conversation Agent")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini")
    graph_builder = build_graph(llm)

    with SqliteSaver.from_conn_string("conversation.db") as checkpointer:

        graph = graph_builder.compile(checkpointer=checkpointer)
        save_graph_image(graph)

        THREAD_ID = "conversation_1"
        config = {"configurable": {"thread_id": THREAD_ID}}

        existing = graph.get_state(config)

        if existing.next:
            print(f"\n  Resuming interrupted conversation (thread: {THREAD_ID})")
            print(f"  Pending node(s): {existing.next}")
            graph.invoke(None, config=config)
        else:
            history = existing.values.get("chat_history", [])
            if history:
                print(f"\n  Continuing saved conversation (thread: {THREAD_ID})")
                print(f"  {len(history)} messages already in history.")
            else:
                print(f"\n  Starting new conversation (thread: {THREAD_ID})")

            initial_state: AgentState = {
                "user_input":   "",
                "should_exit":  False,
                "chat_history": [],
                "llm_response": None,
            }
            graph.invoke(initial_state, config=config)

    print("\n[Session ended -- history saved to conversation.db]")
    print(f"Restart the script to continue as thread '{THREAD_ID}'.")


if __name__ == "__main__":
    main()