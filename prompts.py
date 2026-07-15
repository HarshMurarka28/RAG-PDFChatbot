from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. "
            "Do NOT answer the question, just reformulate it if needed.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the question only from the provided context.

If the answer is not in the context,
say you don't know.

<context>
{context}
</context>
"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)