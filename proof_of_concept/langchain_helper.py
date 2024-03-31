from langchain_community.llms import CTransformers, LlamaCpp
from langchain_google_vertexai import VertexAI
import vertexai
# from huggingface_hub import hf_hub_download
# from llama_cpp import Llama
# from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain, SequentialChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import time
import document_parse_helper as pph
import os
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from operator import itemgetter

load_dotenv()

def get_query_resp(
        question,
        chat_history
    ):

    tic = time.perf_counter()
    
    input_obj = {
        "question": question,
        "chat_history": chat_history
    }

    embedding = pph.EmbeddingStore.load_embeddings('embedding', 'embeddings')

    # special object in langchain that you can use for information retrieval
    # k = 3 means get my three similar documents??
    retriever = embedding.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template =  """
            Your job is to use the following context to answer questions 
            about a how to take care of a baby. If answer does not exist, say that you don't know:

            {context}

            Chat history: {chat_history}

            Question: {question}

            The context is in a particular format but you should NOT mimic the style. 
            You should always answer the question using normal language with professionality.
            Here is the answer:
        """,
        input_variables=[
            "context",
            "chat_history",
            "question"
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=prompt)]
    )

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'hackathon2024-418700-f6f2fc9a356f.json'
    vertexai.init(
        project="hackathon2024-418700",
    )
    
    vertex_llm_text = VertexAI(
        model_name="gemini-pro"
    )

    prompt_chain = (
        {
            "context": itemgetter("question") | retriever,
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question")
        }
        | prompt   
    )

    chain = (
        prompt_chain
        | vertex_llm_text
        | StrOutputParser()
    )

    response = chain.invoke(input_obj)

    toc = time.perf_counter()
    print(f"response time: {toc - tic:0.2f} seconds")
    return response

if __name__ == "__main__":
    pass