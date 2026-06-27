from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
from langchain_groq import ChatGroq
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
import requests

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def get_weather(city: str) -> str:
    """
    Fetches the current weather for a specified city. 
    Use this whenever you need to know the temperature or conditions of a location.
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(data)
        condition = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"In {city}, the weather is currently {condition} with a temperature of {temp}°C."
    
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given stock symbol.

    Args:
        symbol: Stock ticker (e.g. AAPL, TSLA, MSFT)

    Returns:
        Dictionary containing the latest stock information.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv("ALPHAVANTAGE_API_KEY")}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        quote = data.get("Global Quote", {})

        if not quote:
            return {
                "success": False,
                "message": f"No data found for symbol '{symbol.upper()}'."
            }

        return {
            "success": True,
            "symbol": quote["01. symbol"],
            "price": float(quote["05. price"]),
            "change": float(quote["09. change"]),
            "change_percent": quote["10. change percent"],
            "open": float(quote["02. open"]),
            "high": float(quote["03. high"]),
            "low": float(quote["04. low"]),
            "previous_close": float(quote["08. previous close"]),
            "volume": int(quote["06. volume"]),
            "latest_trading_day": quote["07. latest trading day"],
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

tools = [search_tool, get_stock_price, calculator,get_weather]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

