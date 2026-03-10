from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

@tool
def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video by video ID."""
    ytapi = YouTubeTranscriptApi()
    transcript = ytapi.fetch(video_id)
    text = " ".join(t.text for t in transcript)
    return text

@tool
def summarize_transcript(transcript: str) -> str:
    """Summarize a YouTube transcript."""
    prompt = f"Summarize the following transcript:\n\n{transcript}"
    return llm.invoke(prompt).content

@tool
def extract_key_concepts(transcript: str) -> str:
    """Extract key concepts from a transcript."""
    prompt = f"List the key concepts or topics discussed:\n\n{transcript}"
    return llm.invoke(prompt).content

@tool
def generate_quiz(transcript: str) -> str:
    """Generate quiz questions from a transcript."""
    prompt = f"""
    Create 5 quiz questions (with answers) based on this transcript:

    {transcript}
    """
    return llm.invoke(prompt).content

# Create agent with tool
llm = ChatOpenAI(model="gpt-4o-mini")

tools = [
    get_youtube_transcript,
    summarize_transcript,
    extract_key_concepts,
    generate_quiz,
]

llm_with_tools = llm.bind_tools(tools)

def run_agent(user_query: str):
    messages = [
        SystemMessage(content="You are a helpful AI that can analyze YouTube videos using tools."),
        HumanMessage(content=user_query)
    ]

    print(f"User: {user_query}\n")

    for iteration in range(6):
        print(f"--- Iteration {iteration+1} ---")
        response = llm_with_tools.invoke(messages)

        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            messages.append(response)

            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]

                print(f"Tool call: {name}")
                print(f"Args: {args}")

                if name == "get_youtube_transcript":
                    result = get_youtube_transcript.invoke(args)
                elif name == "summarize_transcript":
                    result = summarize_transcript.invoke(args)
                elif name == "extract_key_concepts":
                    result = extract_key_concepts.invoke(args)
                elif name == "generate_quiz":
                    result = generate_quiz.invoke(args)
                else:
                    result = f"Unknown tool: {name}"

                print("Result (truncated):", result[:300], "\n")

                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
        else:
            print(f"Assistant: {response.content}\n")
            return response.content

    return "Max iterations reached"

# Test it
if __name__ == "__main__":
    # Test query that requires tool use
    print("="*60)
    print("TEST 1: Query requiring tool")
    print("="*60)
    run_agent("Get the transcript for the video dQw4w9WgXcQ")

    print("\n" + "="*60)
    print("TEST 2: Query not requiring tool")
    print("="*60)
    run_agent("Say hello!")

    print("\n" + "="*60)
    print("TEST 3: Multiple tool calls")
    print("="*60)
    run_agent("Summarize the transcript of video 1bUy-1hGZpI and generate quiz questions to test my knowledge")