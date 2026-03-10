"""
Tool Calling with LangChain
Extended with calculator, letter counter, and custom tool.
"""

import math
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# ============================================
# PART 1: Define Your Tools
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
    return weather_data.get(location, f"Weather data not available for {location}")


# -------- Calculator Tool --------

from pydantic import BaseModel, Field
from typing import Optional

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
               decimal_places:int = None, radius: float = None,
               length: float = None, width: float = None,
               base: float = None, height: float = None,
               x1: float = None, y1: float = None,
               x2: float = None, y2: float = None) -> float:
    """Perform calculator operations. Use the operation name and provide only the required arguments for that operation."""
    if operation == "sin":
        return math.sin(value)
    elif operation == "cos":
        return math.cos(value)
    elif operation == "tan":
        return math.tan(value)
    elif operation == "add":
        return value + value2
    elif operation == "subtract":
        return value - value2
    elif operation == "multiply":
        return value * value2
    elif operation == "divide":
        return value / value2
    elif operation == "round":
        return round(value, decimal_places)
    elif operation == "circle_area":
        return math.pi * radius ** 2
    elif operation == "rectangle_area":
        return length * width
    elif operation == "triangle_area":
        return 0.5 * base * height
    elif operation == "distance_2d":
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    else:
        raise ValueError(f"Unknown operation: {operation}")

# -------- Letter Count Tool --------

@tool
def count_letter(text: str, letter: str) -> int:
    """Count occurrences of a letter in a text."""
    return text.lower().count(letter.lower())


# -------- Custom Tool (Text Stats) --------

@tool
def text_stats(text: str) -> dict:
    """Return word count and character count for a text."""
    words = text.split()
    return {
        "words": len(words),
        "characters": len(text)
    }


# ============================================
# PART 2: Create LLM with Tools
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini")

tools = [get_weather, calculator, count_letter, text_stats]
llm_with_tools = llm.bind_tools(tools)


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided tools when needed."),
        HumanMessage(content=user_query)
    ]

    print(f"\nUser: {user_query}\n")

    tool_map = {t.name: t for t in tools}

    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        response = llm_with_tools.invoke(messages)

        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            messages.append(response)

            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown tool {function_name}"

                print(f"  Result: {result}")

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))

            print()
        else:
            print(f"Assistant: {response.content}\n")
            return response.content

    return "Max iterations reached"


# ============================================
# PART 4: Test Queries
# ============================================

if __name__ == "__main__":

    print("="*60)
    print("TEST 1: Weather Tool")
    print("="*60)
    run_agent("What's the weather in Tokyo?")

    print("\n" + "="*60)
    print("TEST 2: Multi-call in SAME TURN (letter counts)")
    print("="*60)
    run_agent("Are there more i's than s's in Mississippi riverboats?")

    print("\n" + "="*60)
    print("TEST 3: Sequential Chaining (inner + outer loop)")
    print("="*60)
    run_agent("What is the sin of the difference between the number of i's and s's in Mississippi riverboats?")

    print("\n" + "="*60)
    print("TEST 4: Uses ALL tools in one query")
    print("="*60)
    run_agent(
        "In Mississippi riverboats, count i's and s's, compute the sin of their difference, "
        "tell me the weather in London, and give text statistics."
    )

    print("\n" + "="*60)
    print("TEST 5: Force 5-turn sequential chaining")
    print("="*60)
    run_agent(
        "You must follow each step in order and use tools whenever applicable. "
        "Step 1: Count i's in Mississippi riverboats. "
        "Step 2: Count the number of characters in Mississippi riverboats. "
        "Step 3: Compute the area of a rectangle that has length = # of i's and width = # of characters. "
        "Step 4: Take the cosine of the rectangle's area. "
        "Step 5: Multiply the cosine value by 10 and round to the nearest integer. "
        "Do not skip steps and do not combine steps."
    )
