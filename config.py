import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Models
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "models/gemini-embedding-001"
HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Text Splitting
GEMINI_MAX_CHUNKS = 50
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 4
FETCH_K = 20
LAMBDA_MULT = 0.5