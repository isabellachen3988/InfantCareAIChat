import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import langchain_helper as lh

human_chat_history = []

st.set_page_config(page_title="Infant Care Bot", page_icon="👶")
st.title("Infant Care Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Ask me anything about your baby"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Type your message...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(
        HumanMessage(content=user_query)
    )
    human_chat_history.append(user_query)

    with st.chat_message("Human"):
        st.markdown(user_query)

    # write with AI
    with st.chat_message("AI"):
        response = lh.get_query_resp(
            user_query,
            human_chat_history
        )
        st.write(response)
    st.session_state.chat_history.append(
        AIMessage(content=response)
    )