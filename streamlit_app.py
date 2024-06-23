import streamlit as st

from utils import show_navigation
show_navigation()

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://raft.net/wp-content/uploads/2020/06/logo_vert400-300x192.png");
             background-attachment: fixed;
             background-size: cover;
             opacity: 0.9;
             height: 80%;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#add_bg_from_url() 


st.markdown("# HELLO WORLD")

st.markdown("""
             Please email dan@gprof.com if there are any issues.
             """)