from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import StateGraph, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.pydantic_v1 import BaseModel

from langchain_groq import ChatGroq


class AgentState(TypedDict):
    agent: str
    messages: List[Dict]
    category: str
    sampleResponse: str
    initialResponse: str
    lastUserRequest: str
    analysis: str
    responseToUser: str
    lnode: str

#
# Classes for structured responses from LLM
#
class Category(BaseModel):
    category: str
    sampleResponse: str

class Response(BaseModel):
    response: str

class FinalResponse(BaseModel):
    analysis: str
    response: str

VALID_CATEGORIES=["Product","Raft","SmallTalk","Abuse","Other"]

#
# Main Graph
#
class ChatAnswerer():
    def __init__(self, api_key):
        #self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
        self.model=ChatGroq(model="llama3-70b-8192",api_key=api_key)

        builder = StateGraph(AgentState)
        builder.add_node("initial_classifier", self.initialClassifier)
        builder.add_node("answerProductQuestions", self.answerProductQuestions)
        builder.add_node("answerRaftQuestions", self.answerRaftQuestions)
        builder.add_node("answerSmallTalk", self.answerSmallTalk)
        builder.add_node("answerAbuse", self.answerAbuse)
        builder.add_node("reviewResponse", self.reviewResponse)

        builder.set_entry_point("initial_classifier")
        builder.add_conditional_edges('initial_classifier',self.main_router,
                                      {"Other":"answerSmallTalk", 
                                       "Product":"answerProductQuestions", 
                                       "Raft":"answerRaftQuestions", 
                                       "SmallTalk":"answerSmallTalk",
                                       "Abuse":"answerAbuse",})
        builder.add_edge("answerProductQuestions", "reviewResponse")
        builder.add_edge("answerRaftQuestions", "reviewResponse")
        builder.add_edge("answerSmallTalk", "reviewResponse")
        builder.add_edge("answerAbuse", "reviewResponse")
        builder.add_edge("reviewResponse", END)
        
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(checkpointer = memory)
        self.graph.get_graph().draw_png("graph.png")

    def initialClassifier(self, state: AgentState):
        print("START: initial_classifier")
        my_prompt=f"""
                You are getting requests from users aboutr RAFT.
                Founded in San Jose in 1994, RAFT (Resource Area For Teaching) is a nonprofit providing educators with engaging hands-on 
        learning resources aligned to national content standards.
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
        question=state['lastUserRequest']
        sample_response=state['sampleResponse']
        background_info=""
        my_prompt=f"""
                You are getting requests from users about a specific RAFT product.
                The question is {question}.
                Here is a sample response {sample_response}.
                Here is some useful information about the product.
                {background_info}
                
                Please provide a  response.
                """
        llm_response=self.model.with_structured_output(Response).invoke([
            SystemMessage(content=my_prompt),
            HumanMessage(content=state['lastUserRequest']),
        ])
        return {"initialResponse": llm_response.response }
    
    def answerRaftQuestions(self, state: AgentState):
        print("START: answerRaftQuestions")
        question=state['lastUserRequest']
        sample_response=state['sampleResponse']
        background_info=f"""
        Founded in San Jose in 1994, RAFT (Resource Area For Teaching) is a nonprofit providing educators with engaging hands-on 
        learning resources aligned to national content standards.

        RAFT’s mission is to help parents and educators transform a child’s learning experience through hands-on education. 
        We help bridge the education gap for students and communities by offering engaging STEAM resources to encourage 
        the next generation of innovators, problem solvers, and creators. RAFT does this by providing STEAM Project Kits, 
        Maker Mobile van visits and makerspace builds, STEAM summer camps, tailored educator workshops, and free online 
        learning activity sheets. RAFT’s unique Found Engineering process repurposes donated materials that would have 
        otherwise been sent to the landfill into accessible, low-cost, STEAM Project Kits and assembled by volunteers. 
        Educators, and now parents, can become RAFT members and shop at the warehouse store facility for affordable 
        items devoted to hands-on learning.
        """
        my_prompt=f"""
                You are getting requests from users about RAFT in general.
                The question is {question}.
                
                Here is some useful information about RAFT.
                {background_info}
                
                Please provide a  response.
                """
        llm_response=self.model.with_structured_output(Response).invoke([
            SystemMessage(content=my_prompt),
            HumanMessage(content=state['lastUserRequest']),
        ])
        return {"initialResponse": llm_response.response }
    
    def answerSmallTalk(self, state: AgentState):
        print("START: answerSmallTalk")
        question=state['lastUserRequest']
        sample_response=state['sampleResponse']
        background_info=""
        my_prompt=f"""
                You are getting requests from users.
                The current ask is {question}.
                This question was previously classifies as "small-talk".
                Please respond politely and professionally, but direct the user to questions about RAFT and/or RAFT products.
                Here is a sample response {sample_response}.
                
                Please provide a  response.
                """
        llm_response=self.model.with_structured_output(Response).invoke([
            SystemMessage(content=my_prompt),
            HumanMessage(content=state['lastUserRequest']),
        ])
        return {"initialResponse": llm_response.response }
    
    def answerAbuse(self, state: AgentState):
        print("START: answerAbuse")
        question=state['lastUserRequest']
        sample_response=state['sampleResponse']
        background_info=""
        my_prompt=f"""
                You are getting requests from users.
                The current ask is {question}.
                This question was previously classifies as abusive.
                Please respond politely and professionally, but firmly to dissuade the user from further abusive behavior.
                Here is a sample response {sample_response}.
                
                Please provide a  response.
                """
        llm_response=self.model.with_structured_output(Response).invoke([
            SystemMessage(content=my_prompt),
            HumanMessage(content=state['lastUserRequest']),
        ])
        return {"initialResponse": llm_response.response }


    def reviewResponse(self, state: AgentState):
        print("START: reviewResponse")
        question=state['lastUserRequest']
        initial_response=state['initialResponse']
        category=state['category']
        my_prompt=f"""
                You are an expert at reviewing and revising responses.
                The user's request is :{question}:.
                This request was classified as :{category}:.
                The initial draft of response is :{initial_response}:.
                Please review the response to be professional and courteous.
                Based on this analysis, please provide the final response to be 
                shared with the user.
                
                Please provide both the analysis and the final response.
                """
        llm_response=self.model.with_structured_output(FinalResponse).invoke([
            SystemMessage(content=my_prompt),
            HumanMessage(content=state['lastUserRequest']),
        ])
        return {
            "analysis": llm_response.analysis,
            "responseToUser": llm_response.response,
              }
    
