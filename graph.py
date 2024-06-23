from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

class AgentState(TypedDict):
    agent: str
    messages: List[Dict]
    lastUserRequest: str
    responseToUser: str
    lnode: str


class ChatAnswerer():
    def __init__(self, api_key):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

        builder = StateGraph(AgentState)
        builder.add_node("initial_classifier", self.initialClassifier)
        builder.set_entry_point("initial_classifier")
        builder.add_edge("initial_classifier", END)
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(checkpointer = memory)
        self.graph.get_graph().draw_png("graph.png")

    def initialClassifier(self, state: AgentState):
        print("initial_classifier")
        self.responseToUser = "great job"
        return {
            "lnode": "initial_router", 
            "responseToUser": "success"
        }