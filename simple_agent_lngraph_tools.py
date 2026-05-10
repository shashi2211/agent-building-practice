import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Initialize the Groq Model
model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    groq_api_key=api_key
)

# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import Annotated, TypedDict, add_messages

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

# create tools
tool = TavilySearchResults(max_results=2)
# rest = tool.invoke("What is the capital of france?")
# print(rest)

tools = [tool]

model_with_tools = model.bind_tools(tools)

# res = model_with_tools.invoke("What's a node in langgraph?")
# print(res)


# message in the state and calls tools if the message contains tool_calls
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition


def bot(state: State):
    # print(state.items())
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)

# instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)  # Add the node to the graph

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)

# STEP 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")

# ADD MEMORY NODE
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# 1. Initialize the Saver
saver = SqliteSaver.from_conn_string(":memory:")


with saver as memory:
    # 3. Compile the graph INSIDE the block
    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}}

    print("--- Chatbot Started (Type 'q' to quit) ---")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        # 4. Stream inside the block while the connection is alive
        for event in graph.stream(
            {"messages": [("user", user_input)]}, 
            config, 
            stream_mode="values"
        ):
            # In 'values' mode, event is the full state dictionary
            if event["messages"]:
                last_message = event["messages"][-1]
                if isinstance(last_message, BaseMessage) and last_message.type == "ai":
                    print("Assistant:", last_message.content)