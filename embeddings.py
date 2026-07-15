from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    GEMINI_MAX_CHUNKS,
    HUGGINGFACE_EMBEDDING_MODEL
)

def get_embeddings(chunk_count):

    if chunk_count <= GEMINI_MAX_CHUNKS:

        print("Using Gemini Embeddings")

        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

    print("Using HuggingFace BGE Embeddings")
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
    model_name=HUGGINGFACE_EMBEDDING_MODEL
)