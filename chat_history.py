from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

def get_chat_history(chat_history):

    history = []

    for msg in chat_history:

        if msg["role"] == "user":
            history.append(
                HumanMessage(content=msg["content"])
            )

        else:
            history.append(
                AIMessage(content=msg["content"])
            )

    return history