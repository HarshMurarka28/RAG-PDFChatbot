from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL
import streamlit as st
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL
    )