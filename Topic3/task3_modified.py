"""
Manual Tool Calling Exercise
Students will see how tool calling works under the hood.
Now extended with a calculator tool for geometric functions.
"""

import json
import math
from openai import OpenAI

# ============================================
# PART 1: Define Your Tools
# ============================================

def get_weather(location: str) -> str:
    """Get the current weather for a location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


# ---- Calculator Geometric Functions ----

def circle_area(radius: float) -> float:
    return math.pi * radius ** 2

def rectangle_area(length: float, width: float) -> float:
    return length * width

def triangle_area(base: float, height: float) -> float:
    return 0.5 * base * height

def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculator(operation: str, parameters: dict) -> str:
    """
    Unified calculator tool dispatcher for geometric operations.
    """
    try:
        if operation == "circle_area":
            return str(circle_area(**parameters))
        elif operation == "rectangle_area":
            return str(rectangle_area(**parameters))
        elif operation == "triangle_area":
            return str(triangle_area(**parameters))
        elif operation == "distance_2d":
            return str(distance_2d(**parameters))
        else:
            return f"Error: Unknown operation '{operation}'"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ============================================
# PART 2: Describe Tools to the LLM
# ============================================

tools = [
    # Weather tool
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    }
                },
                "required": ["location"]
            }
        }
    },

    # Calculator tool
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform geometric calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["circle_area", "rectangle_area", "triangle_area", "distance_2d"]
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "radius": {"type": "number"},
                            "length": {"type": "number"},
                            "width": {"type": "number"},
                            "base": {"type": "number"},
                            "height": {"type": "number"},
                            "x1": {"type": "number"},
                            "y1": {"type": "number"},
                            "x2": {"type": "number"},
                            "y2": {"type": "number"}
                        }
                    }
                },
                "required": ["operation", "parameters"]
            }
        }
    }
]


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """

    client = OpenAI()

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": user_query}
    ]

    print(f"User: {user_query}\n")

    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                # Manual dispatch
                if function_name == "get_weather":
                    result = get_weather(**function_args)
                elif function_name == "calculator":
                    result = calculator(**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })

            print()
        else:
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content

    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("TEST 1: Weather tool")
    print("="*60)
    run_agent("What's the weather like in San Francisco?")

    print("\n" + "="*60)
    print("TEST 2: Greeting")
    print("="*60)
    run_agent("Say hello!")

    print("\n" + "="*60)
    print("TEST 3: Multiple weather calls")
    print("="*60)
    run_agent("What's the weather in New York and London?")

    print("\n" + "="*60)
    print("TEST 4: Calculator - circle area")
    print("="*60)
    run_agent("What is the area of a circle with radius 3?")

    print("\n" + "="*60)
    print("TEST 5: Calculator - distance")
    print("="*60)
    run_agent("Find the distance between (0,0) and (3,4).")
