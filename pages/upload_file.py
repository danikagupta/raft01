import streamlit as st

from utils import show_navigation
show_navigation()


import os
import PyPDF2
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import hashlib
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']
#PINECONE_API_ENV=st.secrets['PINECONE_API_ENV']
PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']
PINECONE_CLOUD_NAME=st.secrets['PINECONE_CLOUD_NAME']
PINECONE_REGION_NAME=st.secrets['PINECONE_REGION_NAME']



client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

def pdf_to_text(uploaded_file):
    pdfReader = PyPDF2.PdfReader(uploaded_file)
    count = len(pdfReader.pages)
    text=""
    for i in range(count):
        page = pdfReader.pages[i]
        text=text+page.extract_text()
    return text

def embed(text,filename):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud=PINECONE_CLOUD_NAME,region=PINECONE_REGION_NAME)
    index = pc.Index(PINECONE_INDEX_NAME,spec=spec)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  = 200,length_function = len,is_separator_regex = False)
    docs=text_splitter.create_documents([text])
    for idx,d in enumerate(docs):
        hash=hashlib.md5(d.page_content.encode('utf-8')).hexdigest()
        embedding=client.embeddings.create(model="text-embedding-ada-002", input=d.page_content).data[0].embedding
        metadata={"hash":hash,"text":d.page_content,"index":idx,"model":"text-embedding-ada-003","docname":filename}
        index.upsert([(hash,embedding,metadata)])
    return

#
# Direcly access Text Input    
#
st.markdown("Upload text directly")
uploaded_text = st.text_area("Enter Text","")
if st.button('Process and Upload Text'):
    embedding = embed(uploaded_text,"Anonymous")
#
# Accept a PDF file using Streamlit
# Upload to Pinecone
#
st.markdown("# Upload file: PDF")
uploaded_file=st.file_uploader("Upload PDF file",type="pdf")
if uploaded_file is not None:
    if st.button('Process and Upload File'):
        pdf_text = pdf_to_text(uploaded_file)
        embedding = embed(pdf_text,uploaded_file.name)