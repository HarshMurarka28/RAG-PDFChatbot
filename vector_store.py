import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from langchain_community.vectorstores import FAISS

from embeddings import get_embeddings

def save_uploaded_file(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.getvalue())

        return tmp_file.name

def load_documents(pdf_path):

    loader = PyPDFLoader(pdf_path)

    return loader.load()

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP

    )
    return splitter.split_documents(documents)

def create_vector_store(documents):
    try:
        embeddings = get_embeddings(len(documents))
    except Exception as e:
        st.error(
    "Failed to generate embeddings. This may be due to API quota limits, an internet issue, or an unsupported PDF."
)
        return None
    
    return FAISS.from_documents(
        documents,
        embeddings
    )

def build_vector_store(uploaded_file):

    pdf_path = None

    try:

        pdf_path = save_uploaded_file(uploaded_file)

        docs = load_documents(pdf_path)

        chunks = split_documents(docs)

        return create_vector_store(chunks)

    except Exception as e:

        st.error(
            "Failed to process the uploaded PDF."
        )

        st.exception(e)

        return None

    finally:

        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)