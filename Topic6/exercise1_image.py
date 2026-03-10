import base64
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage

import ollama
from IPython.display import display
import ipywidgets as widgets
from PIL import Image
import io
from google.colab import files
import gradio as gr


class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    user_input: str
    image_b64: str | None
    should_exit: bool

from PIL import Image
import io, base64

def prepare_image_for_llava(image_bytes,
                            max_dim=128,
                            max_megapixels=0.1,
                            jpeg_quality=70):
    """
    Resize + compress image so Ollama LLaVA won't crash.
    Returns base64 string.
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ---------- Step 1: limit megapixels ----------
    w, h = img.size
    megapixels = (w*h)/1_000_000

    if megapixels > max_megapixels:
        scale = (max_megapixels/megapixels) ** 0.5
        img = img.resize(
            (int(w*scale), int(h*scale)),
            Image.LANCZOS
        )

    # ---------- Step 2: limit dimension ----------
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    # ---------- Step 3: compress JPEG ----------
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)

    # ---------- Step 4: convert to base64 ----------
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    print(f"Final image size: {img.size}")

    return img_b64

def route_after_input(state):

    if state.get("should_exit"):
        return END

    if state.get("user_input","") == "":
        return "get_user_input"

    return "call_llava"

def call_llava(state: AgentState):

    text = state["user_input"]
    img = state["image_b64"]

    # build conversation history
    msgs = []

    for m in state["messages"][:-1]:
        role = "assistant" if isinstance(m, AIMessage) else "user"
        msgs.append({"role":role, "content":m.content})

    # current user message WITH IMAGE
    msgs.append({
        "role":"user",
        "content": text,
        "images":[img]
    })

    response = ollama.chat(
        model="llava:7b-v1.5-q4_0",
        messages=msgs
    )

    answer = response["message"]["content"]

    return {"messages":[AIMessage(content=answer)]}

def create_graph():

    graph = StateGraph(AgentState)

    graph.add_node("call_llava", call_llava)

    graph.add_edge(START, "call_llava")
    graph.add_edge("call_llava", END)

    return graph.compile()

graph = create_graph()

# Global conversation state
state = {
    "messages":[
        SystemMessage(
            content="You are a helpful vision assistant. Answer questions about the image clearly and concisely."
        )
    ],
    "user_input":"",
    "image_b64":None,
    "should_exit":False
}

def chat_with_image(user_message, image, chat_history):

    global state

    # If image uploaded, process it
    if image is not None:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = prepare_image_for_llava(buffered.getvalue())
        state["image_b64"] = img_b64

    if state["image_b64"] is None:
        return chat_history + [("Please upload an image first.", "")]

    # Update state
    state["user_input"] = user_message
    state["messages"].append(HumanMessage(content=user_message))

    try:
        result = graph.invoke(state)
        assistant_reply = result["messages"][-1].content

    except Exception as e:
        assistant_reply = f"Error: {str(e)}"

    state["messages"].append(AIMessage(content=assistant_reply))

    chat_history.append((user_message, assistant_reply))

    return chat_history


with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Vision-Language Chat Agent (LLaVA + LangGraph)")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

    chatbot = gr.Chatbot()

    msg = gr.Textbox(label="Ask a question about the image")
    send_btn = gr.Button("Send")

    send_btn.click(
        chat_with_image,
        inputs=[msg, image_input, chatbot],
        outputs=chatbot
    )

demo.launch(share=True)
