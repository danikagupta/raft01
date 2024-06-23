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
    category: str
    sampleResponse: str
    lastUserRequest: str
    responseToUser: str
    lnode: str

#
# Classes for structured responses from LLM
#
class Category(BaseModel):
    category: str
    sampleResponse: str

VALID_CATEGORIES=["Product","Raft","SmallTalk","Abuse","Other"]

#
# Main Graph
#
class ChatAnswerer():
    def __init__(self, api_key):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

        builder = StateGraph(AgentState)
        builder.add_node("initial_classifier", self.initialClassifier)
        builder.add_node("answerProductQuestions", self.answerProductQuestions)
        builder.add_node("answerRaftQuestions", self.answerRaftQuestions)
        builder.add_node("answerSmallTalk", self.answerSmallTalk)
        builder.add_node("answerAbuse", self.answerAbuse)

        builder.set_entry_point("initial_classifier")
        builder.add_conditional_edges('initial_classifier',self.main_router,
                                      {"Other":END, 
                                       "Product":"answerProductQuestions", 
                                       "Raft":"answerRaftQuestions", 
                                       "SmallTalk":"answerSmallTalk",
                                       "Abuse":"answerAbuse",})
        builder.add_edge("answerProductQuestions", END)
        builder.add_edge("answerRaftQuestions", END)
        builder.add_edge("answerSmallTalk", END)
        builder.add_edge("answerAbuse", END)
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

        return {
            "lnode": "initialClassifier", 
            #"responseToUser": f"Category: {category}\n Sample Response: {sampleResponse}",
            "category": category,
            "sampleResponse": sampleResponse,
        }
    
    def main_router(self, state: AgentState):
        my_category=state['category']
        print(f"\n\nSTART: mainRouter with msg {state['lastUserRequest']} and category {my_category}")
        if my_category in VALID_CATEGORIES:
            return my_category
        else:
            print(f"Unknown category {my_category}")
            return END
        
    def answerProductQuestions(self, state: AgentState):
        print("START: answerProductQuestions")
        return {"responseToUser": state['sampleResponse']}
    
    def answerRaftQuestions(self, state: AgentState):
        print("START: answerRaftQuestions")
        return {"responseToUser": state['sampleResponse']}
    
    def answerSmallTalk(self, state: AgentState):
        print("START: answerSmallTalk")
        return {"responseToUser": state['sampleResponse']}
    
    def answerAbuse(self, state: AgentState):
        print("START: answerAbuse")
        return {"responseToUser": state['sampleResponse']}
    
    def answerOther(self, state: AgentState):
        print("START: answerOther")
        return {"responseToUser": state['sampleResponse']}
    
    def answerUnknown(self, state: AgentState):
        print("START: answerUnknown")
        return {"responseToUser": "I am sorry, I do not understand your request."}
    
