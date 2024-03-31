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
        chat_history,
        use_backup=False
    ):

    tic = time.perf_counter()
    
    input_obj = {
        "question": question,
        "chat_history": chat_history
    }

    embedding_name = 'embedding' if not use_backup else 'embedding_small'

    embedding = pph.EmbeddingStore.load_embeddings(embedding_name, 'embeddings')

    # special object in langchain that you can use for information retrieval
    # k = 3 means get my three similar documents??
    retriever = embedding.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template =  
"""
You are a professional expertise with acurate knowledge on infant cares.
Your job is to use the following context to answer questions about a how to take care of a baby.
The context provided are from academic researches and are highly reliable. 
You should try to stick to the context as much as possible.
The context is in a particular format but you should NOT mimic the style. 
You should always answer the question using normal language with professionality.
The user are new parents and may not have professional knowledge. So you can try to paraphrase the answer to make it more understandable to the user.
In the end of your answer, include ONLY the file name and page number of the source. For example: "Source: some_file.pdf, page 5".
Context: {context}
Chat history: {chat_history}
Question: {question}
Here is the answer:
""",
        input_variables=["context", "chat_history", "question"]
    )

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=prompt)]
    )

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'hackathon2024-418700-0ae5baa3a912.json'
    vertexai.init(
        project="hackathon2024-418700",
    )
    
    vertex_llm_text = VertexAI(
        model_name="gemini-pro",
        temperature=0.9
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
    print(f'Question: {question}')
    print(f'Chat history: {chat_history}')
    print(retriever.get_relevant_documents(question))
    print(f"response time: {toc - tic:0.2f} seconds")
    print(response)
    return response

if __name__ == "__main__":
    pass