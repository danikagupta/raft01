import streamlit as st

def show_navigation():
    st.sidebar.image("https://raft.net/wp-content/uploads/2020/06/logo_vert400-300x192.png", width=200)  
    with st.container(border=True):
        col1,col2,col3,col4=st.columns(4)
        col1.page_link("streamlit_app.py", label="Home", icon="🏠")
        col2.page_link("pages/admin.py", label="Admin", icon="1️⃣")
        col3.page_link("pages/rag.py", label="Chat", icon="2️⃣")
        col4.page_link("pages/upload_file.py", label="Upload", icon="3️⃣")
