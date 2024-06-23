from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import StateGraph, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.pydantic_v1 import BaseModel


class AgentState(TypedDict):
    agent: str
    messages: List[Dict]
    lastUserRequest: str
    responseToUser: str
    lnode: str

#
# Classes for structured responses from LLM
#
class Category(BaseModel):
    category: str
    sampleResponse: str


#
# Main Graph
#
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
        print("START: initial_classifier")
        my_prompt=f"""
                You are getting requests from users aboutr RAFT.
                Please classify the request.
                
                If the request is question about a specific product, category is 'Product'
                If the request is a general question about RAFT, category is 'Raft'
                If the request is a general, pleasant conversation, category is 'SmallTalk'
                If the request is abusive or offensive, category is 'Abuse'
                Else the category is 'Other'
                
                Please also provide a sample response.
                """
        llm_response=self.model.with_structured_output(Category).invoke([
            SystemMessage(content=my_prompt),
            HumanMessage(content=state['lastUserRequest']),
        ])
        category = llm_response.category
        sampleResponse = llm_response.sampleResponse
        print(f"Category: {category}")
        print(f"Sample Response: {sampleResponse}")

        self.responseToUser = "great job"
        return {
            "lnode": "initial_router", 
            "responseToUser": f"Category: {category}\n Sample Response: {sampleResponse}"
        }