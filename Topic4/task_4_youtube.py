from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
import asyncio
import time
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


@tool
def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video by video ID."""
    ytapi = YouTubeTranscriptApi()
    transcript = ytapi.fetch(video_id)
    text = " ".join(t.text for t in transcript)
    return text
    # transcript = YouTubeTranscriptApi.get_transcript(video_id)
    # return " ".join([entry['text'] for entry in transcript])

tools = [get_youtube_transcript]

class ConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    verbose: bool
    command: str
    transcript: str  # <-- NEW

def store_transcript_node(state: ConversationState) -> ConversationState:
    """
    Extract last tool output and store transcript in state.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "get_youtube_transcript":
            return {"transcript": msg.content}
    return {}


def input_node(state: ConversationState) -> ConversationState:
    """
    Get input from the user and add it to the conversation.
    
    This node:
    - Prompts the user for input
    - Handles special commands (quit, exit, verbose, quiet)
    - Adds user message to conversation history (for real messages only)
    - Sets command field for special commands
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with new user message or command
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: input_node")
        print("="*80)
    
    # Get user input
    user_input = input("\nYou: ").strip()
    
    # Handle exit commands
    if user_input.lower() in ["quit", "exit"]:
        if state.get("verbose", True):
            print("[DEBUG] Exit command received")
        # Set command field, don't add to messages
        return {"command": "exit"}
    
    # Handle verbose toggle
    if user_input.lower() == "verbose":
        print("[SYSTEM] Verbose mode enabled")
        # Set command field and update verbose flag
        return {"command": "verbose", "verbose": True}
    
    if user_input.lower() == "quiet":
        print("[SYSTEM] Verbose mode disabled")
        # Set command field and update verbose flag
        return {"command": "quiet", "verbose": False}
    
    # Add user message to conversation history
    if state.get("verbose", True):
        print(f"[DEBUG] User input: {user_input}")
    
    # Clear command field and add message
    return {"command": None, "messages": [HumanMessage(content=user_input)]}


def call_react_agent(state: ConversationState) -> ConversationState:
    global react_agent
    
    messages = list(state["messages"])

    # Inject transcript as system context
    if "transcript" in state and state["transcript"]:
        messages = [
            SystemMessage(content=f"VIDEO TRANSCRIPT:\n{state['transcript'][:12000]}"),
            *messages
        ]

    result = react_agent.invoke({"messages": messages})
    new_messages = result["messages"][len(messages):]
    return {"messages": new_messages}


def output_node(state: ConversationState) -> ConversationState:
    """
    Display the assistant's final response to the user.
    
    This node:
    - Extracts the last AI message from the conversation
    - Prints it to the console
    - Returns empty dict (no state changes)
    
    Args:
        state: Current conversation state
        
    Returns:
        Empty dict (no state modifications)
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: output_node")
        print("="*80)
    
    # Find the last AI message in the conversation
    # (there may be tool messages mixed in)
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_message = msg
            break
    
    if last_ai_message:
        print(f"\nAssistant: {last_ai_message.content}")
    else:
        print("\n[WARNING] No assistant response found")
    
    return {}


def trim_history(state: ConversationState) -> ConversationState:
    """
    Manage conversation history length to prevent unlimited growth.
    
    Strategy:
    - Keep the system message (if present)
    - Keep the most recent 100 messages
    - This allows ~50 conversation turns (user + assistant pairs)
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with trimmed message history (if needed)
    """
    messages = state["messages"]
    max_messages = 100
    
    # Only trim if we've exceeded the limit
    if len(messages) > max_messages:
        if state.get("verbose", True):
            print(f"\n[DEBUG] History length: {len(messages)} messages")
            print(f"[DEBUG] Trimming to most recent {max_messages} messages")
        
        # Preserve system message if it exists at the start
        if messages and isinstance(messages[0], SystemMessage):
            # Keep system message + last (max_messages - 1) messages
            trimmed = [messages[0]] + list(messages[-(max_messages - 1):])
            if state.get("verbose", True):
                print(f"[DEBUG] Preserved system message + {max_messages - 1} recent messages")
        else:
            # Just keep the last max_messages
            trimmed = list(messages[-max_messages:])
            if state.get("verbose", True):
                print(f"[DEBUG] Kept {max_messages} most recent messages")
        
        return {"messages": trimmed}
    
    # No trimming needed
    return {}


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_after_input(state: ConversationState) -> Literal["call_react_agent", "end", "input"]:
    """
    Determine where to route after input based on command field.
    
    Logic:
    - If command is "exit", route to END
    - If command is "verbose" or "quiet", route back to input
    - Otherwise (command is None), route to the ReAct agent
    
    Args:
        state: Current conversation state
        
    Returns:
        "end" to terminate, "input" for verbose toggle, "call_react_agent" to continue
    """
    command = state.get("command")
    
    # Check for exit command
    if command == "exit":
        if state.get("verbose", True):
            print("[DEBUG] Routing to END (exit requested)")
        return "end"
    
    # Check for verbose toggle commands - route back to input
    if command in ["verbose", "quiet"]:
        if state.get("verbose", True):
            print("[DEBUG] Routing back to input (verbose toggle)")
        return "input"
    
    # Normal message - route to agent
    if state.get("verbose", True):
        print("[DEBUG] Routing to call_react_agent")
    return "call_react_agent"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# Global variable to hold the ReAct agent
react_agent = None

def create_conversation_graph():
    """
    Build the conversation graph with persistent multi-turn capability.
    
    Graph structure (single conversation with looping):
    
        Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
        Ã¢â€â€š                                                      Ã¢â€â€š
        Ã¢â€“Â¼                                                      Ã¢â€â€š
      input_node Ã¢â€â‚¬Ã¢â€â‚¬(check command)Ã¢â€â‚¬Ã¢â€â‚¬> call_react_agent        Ã¢â€â€š
          Ã¢â€“Â²                              Ã¢â€â€š                     Ã¢â€â€š
          Ã¢â€â€š                              Ã¢â€“Â¼                     Ã¢â€â€š
          Ã¢â€â€š                         output_node                Ã¢â€â€š
          Ã¢â€â€š                              Ã¢â€â€š                     Ã¢â€â€š
          Ã¢â€â€š                              Ã¢â€“Â¼                     Ã¢â€â€š
          Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬(verbose/quiet)       trim_history Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
          
          Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬(exit)Ã¢â€â‚¬Ã¢â€â‚¬> END
                                         
    Key features:
    - Single conversation maintained in state.messages
    - Command field used for special commands (no sentinel messages!)
    - Graph loops back to input_node after each turn
    - Verbose/quiet commands route directly back to input
    - History automatically trimmed when it grows too long
    - No Python loops or checkpointing needed
    
    Returns:
        Compiled LangGraph application
    """
    global react_agent
    
    # ========================================================================
    # Create the ReAct Agent
    # ========================================================================
    
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7
    )
    
    # System message to encourage tool usage
    system_message = """
      You are a learning assistant.

      If the user requests:
      - summary: summarize the transcript in bullets
      - key concepts: extract important ideas
      - quiz: generate questions
      - grade quiz: evaluate answers using transcript context

      If transcript is missing, call get_youtube_transcript first.
    """
    
    # Create the ReAct agent using the built-in function
    # This agent handles the thought/action/observation loop internally
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_message
    )
    
    print("[SYSTEM] ReAct agent created successfully")
    
    # ========================================================================
    # Create the Conversation Wrapper Graph
    # ========================================================================
    
    workflow = StateGraph(ConversationState)
    
    # Add all nodes
    workflow.add_node("input", input_node)
    workflow.add_node("call_react_agent", call_react_agent)
    workflow.add_node("output", output_node)
    workflow.add_node("trim_history", trim_history)
    workflow.add_node("store_transcript", store_transcript_node)
    
    # Set entry point - conversation always starts at input
    workflow.set_entry_point("input")
    
    # Add conditional edge from input based on command field
    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {
            "call_react_agent": "call_react_agent",
            "input": "input",  # Loop back for verbose/quiet
            "end": END
        }
    )
    
    # Add linear edges for the main conversation flow
    # Agent -> Output -> Trim -> Input (loops back!)
    workflow.add_edge("call_react_agent", "store_transcript")
    workflow.add_edge("store_transcript", "output")
    workflow.add_edge("output", "trim_history")
    workflow.add_edge("trim_history", "input")  # This creates the loop!
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_graphs(wrapper_app):
    """
    Generate Mermaid diagrams for both graphs.
    
    Creates:
    - langchain_react_agent.png: Internal ReAct agent (thought/action/observation)
    - langchain_conversation_graph.png: Conversation loop wrapper
    
    Args:
        wrapper_app: Compiled conversation graph
    """
    global react_agent
    
    # Visualize the ReAct agent
    try:
        react_png = react_agent.get_graph().draw_mermaid_png()
        with open("langchain_react_agent.png", "wb") as f:
            f.write(react_png)
        print("[SYSTEM] ReAct agent graph saved to 'langchain_react_agent.png'")
    except Exception as e:
        print(f"[WARNING] Could not generate ReAct agent visualization: {e}")
    
    # Visualize the conversation wrapper
    try:
        wrapper_png = wrapper_app.get_graph().draw_mermaid_png()
        with open("langchain_conversation_graph.png", "wb") as f:
            f.write(wrapper_png)
        print("[SYSTEM] Conversation graph saved to 'langchain_conversation_graph.png'")
    except Exception as e:
        print(f"[WARNING] Could not generate conversation graph visualization: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """
    Main execution function.
    
    This function:
    1. Creates the conversation graph
    2. Visualizes the graph structure
    3. Initializes the conversation state
    4. Invokes the graph ONCE
    
    The graph then runs indefinitely via internal looping (trim_history -> input)
    until the user types 'quit' or 'exit'.
    """
    print("="*80)
    print("LangGraph ReAct Agent - Persistent Multi-Turn Conversation")
    print("="*80)
    print("\nThis system uses create_react_agent with graph-based looping:")
    print("  - Single persistent conversation across all turns")
    print("  - History managed automatically (trimmed after 100 messages)")
    print("  - Loops via graph edges (no Python loops or checkpointing)")
    print("\nCommands:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'verbose' to enable detailed tracing")
    print("  - Type 'quiet' to disable detailed tracing")
    print("="*80)
    
    # Create the conversation graph
    app = create_conversation_graph()
    
    # Visualize both graphs
    visualize_graphs(app)
    
    # Initialize conversation state
    # This state persists across all turns via graph looping
    initial_state = {
        "messages": [],
        "verbose": True,
        "command": None
    }
    
    print("\n[SYSTEM] Starting conversation...\n")
    
    try:
        # Invoke the graph ONCE
        # The graph will loop internally until user exits
        # Each iteration: input -> agent -> output -> trim -> input (loop!)
        # Verbose commands: input -> input (direct loop!)
        await app.ainvoke(initial_state)
        
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Interrupted by user (Ctrl+C)")
    
    print("\n[SYSTEM] Conversation ended. Goodbye!\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        # Running inside Jupyter
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())