import streamlit as st
import os
from graph import ChatAnswerer
import random

from utils import show_navigation
show_navigation()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGSMITH_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']

DEBUGGING=0

def start_chat():
    st.title('RAFT Customer Service Chatbot')
    avatars={"system":"💻","user":"🧑‍💼","assistant":"🧠"}

    if "messages" not in st.session_state:
        st.session_state.messages = []

    #
    # Keeping context of conversations.
    # In practice, this will be say from the Slack - perhaps hash of user-id and channel-id.
    #
    if "thread-id" not in st.session_state:
        st.session_state.thread_id = random.randint(1000, 9999)
    thread_id = st.session_state.thread_id

    # Reminder
    st.sidebar.write("""
    Use cases:
    1. Questions about RAFT
    2. Questions about a specific product
    3. General conversation
    4. Abusive or irrelevant content
    5. Anything else
                      """)


    for message in st.session_state.messages:
        if message["role"] != "system":
            avatar=avatars[message["role"]]
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=avatars["user"]):
            st.markdown(prompt)
        #abot=ChatAnswerer(st.secrets['OPENAI_API_KEY'])
        abot=ChatAnswerer(st.secrets['GROQ_API_KEY'])
        thread={"configurable":{"thread_id":thread_id}}
        for s in abot.graph.stream({'lastUserRequest':prompt,'messages':st.session_state.messages},thread):
            if DEBUGGING:
                print(f"GRAPH RUN: {s}")
                st.write(s)
            for k,v in s.items():
                if DEBUGGING:
                    print(f"Key: {k}, Value: {v}")
                if resp := v.get("responseToUser"):
                    with st.chat_message("assistant", avatar=avatars["assistant"]):
                        st.write(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

if __name__ == '__main__':
    start_chat()