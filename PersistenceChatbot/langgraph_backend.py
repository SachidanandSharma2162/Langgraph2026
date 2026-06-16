from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage,SystemMessage, HumanMessage
from IPython.display import Image
from pydantic import BaseModel,Field
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
import operator
from langgraph.checkpoint.memory import InMemorySaver # for Persistence

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)