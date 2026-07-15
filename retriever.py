from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_classic.chains.retrieval import (
    create_retrieval_chain
)
from prompts import (
    qa_prompt,
    contextualize_q_prompt,
)

from config import (
    TOP_K,
    FETCH_K,
    LAMBDA_MULT,
)

def get_retriever(vectors):

    return vectors.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K,
            "lambda_mult": LAMBDA_MULT,
        }
    )

def get_history_aware_retriever(
    llm,
    retriever,
):

    return create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

def get_retrieval_chain(
    llm,
    vectors,
):
    document_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

    retriever = get_retriever(vectors)

    history_aware = get_history_aware_retriever(
    llm,
    retriever
)

    return create_retrieval_chain(
    history_aware,
    document_chain
)
