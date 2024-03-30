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
from enum_helper import Tone

from operator import itemgetter

load_dotenv()

# temperature: randomness of the outcome or how creative you want your model to be
def get_query_resp(
        question,
        chat_history,
        tone
    ):

    tic = time.perf_counter()
    embedding = pph.EmbeddingStore.load_embeddings('embedding', 'embeddings')

    # special object in langchain that you can use for information retrieval
    # k = 3 means get my three similar documents??
    retriever = embedding.as_retriever(search_kwargs={"k": 3})
    # retriever = embedding.as_retriever()
    tone_template = ""
    if (tone == Tone.FATHER_SPEAK):
        tone_template = """
            Your job is to use the following context to answer questions
            to a father about how to take care of a baby. 
            Please use references to to motor vehicles:
        """
    elif (tone == Tone.MOTHER_SPEAK):
        tone_template = """
            Your job is to use the following context to answer questions
            to a mother about how to take care of a baby. 
            Please use references that a mother would understand:
        """

    prompt = PromptTemplate(
        template =  """
            Your job is to use the following context to answer questions 
            about a how to take care of a baby:

            {context}

            Chat history: {chat_history}

            Question: {question}

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

    input_obj = {
        "question": question,
        "chat_history": chat_history,
        "tone": tone_template
    }

    if (tone == Tone.DEFAULT):
        response = chain.stream(input_obj)
    else:
        tone_prompt = ChatPromptTemplate.from_template(
            """
            {tone}

            {information}
            """
        )

        tone_prompt_chain = (
            {
                "information": chain,
                "tone": itemgetter("tone")
            }
            | tone_prompt
        )

        print(tone_prompt_chain.invoke(input_obj))

        tone_chain = (
            tone_prompt_chain
            | vertex_llm_text
            | StrOutputParser()
        )

        response = tone_chain.stream(input_obj)

    toc = time.perf_counter()
    print(f"response time: {toc - tic:0.2f} seconds")
    return response

if __name__ == "__main__":
    pass