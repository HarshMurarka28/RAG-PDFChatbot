import streamlit as st
from chat_history import get_chat_history
from retriever import get_retrieval_chain
from llm import get_llm
from vector_store import build_vector_store


st.title("Gemma model document Q&A")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = get_llm()

uploaded_file = st.file_uploader(
    "Upload your file",
    type="pdf"
)


if uploaded_file:

    if (
    "current_file" not in st.session_state
    or st.session_state.current_file != uploaded_file.name
):
        with st.spinner("Processing your PDF..."):

            st.session_state.vectors = build_vector_store(
            uploaded_file
    )
        if st.session_state.vectors is None:
            st.stop()
        st.session_state.current_file = uploaded_file.name
        st.session_state.chat_history = []

    st.success("PDF uploaded successfully!")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt1 = st.chat_input("Ask something about your PDF")

if prompt1:
    
    if "vectors" not in st.session_state:
        st.warning("Please upload a PDF first.")
        st.stop()

    retriever_chain = get_retrieval_chain(
    llm,
    st.session_state.vectors,
)
    with st.chat_message("user"):
        st.markdown(prompt1)
    try:

        response = retriever_chain.invoke(
        {
            "input": prompt1,
            "chat_history": get_chat_history(
                st.session_state.chat_history
            ),
        }
    )

    except Exception as e:

        st.error(
        "Failed to generate a response. Please try again."
    )

        st.exception(e)

        st.stop()
    st.session_state.chat_history.append(
        {
        "role": "user",
        "content": prompt1
        }
    )
    with st.chat_message("assistant"):
        st.markdown(response["answer"])

    st.session_state.chat_history.append(
        {
        "role": "assistant",
        "content": response["answer"]
        }
    )
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"], 1):
            st.markdown(f"### Source {i}")
            st.write(f"**Page:** {doc.metadata.get('page', 0) + 1}")
            st.write(doc.page_content)
            st.divider()