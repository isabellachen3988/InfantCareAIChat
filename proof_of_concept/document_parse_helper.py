from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredHTMLLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from abc import ABC, abstractmethod

class EmbeddingStore(ABC):
    @staticmethod
    def _save_pickle(vector_store, store_name, path):
        with open(f"{path}/faiss_{store_name}.pk1", "wb") as f:
            pickle.dump(vector_store, f)
    
    @abstractmethod
    def store_embeddings(self, embeddings, store_name, path):
        pass
    
    @staticmethod
    def load_embeddings(store_name, path):
        with open(f"{path}/faiss_{store_name}.pk1", "rb") as f:
            vector_store = pickle.load(f)
            return vector_store

    @staticmethod
    def _get_embeddings():
        # create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device':"cpu"}
        )

        # embeddings = HuggingFaceEmbeddings(
        #     model_name = "hkunlp/instructor-xl",
        #     model_kwargs={'device': "cuda"}
        # )

        return embeddings

class DocumentEmbeddingStore(EmbeddingStore):
    @abstractmethod
    def store_embeddings(self, docs, store_name, path):
        pass

    def _store_embeddings_helper(self, loader, store_name, path):
        docs = loader.load()

        # split text/documents into chunks
        # chunk size means the number of characters
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(docs)

        vector_store = FAISS.from_documents(docs, self._get_embeddings())
        self._save_pickle(vector_store, store_name, path)

class TextEmbeddingStore(EmbeddingStore):
    def store_embeddings(self, store_name, path):
        vector_store = FAISS.from_texts(["blue cats are generally named thomas"], self._get_embeddings())
        self._save_pickle(vector_store, store_name, path)

class PDFEmbeddingStore(DocumentEmbeddingStore):
    def store_embeddings(self, store_name, path):
        loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)

        self._store_embeddings_helper(loader, store_name, path)

class WebEmbeddingStore(DocumentEmbeddingStore):
    def store_embeddings(self, store_name, path):
        loader = WebBaseLoader("https://warriors.fandom.com/wiki/Names")

        self._store_embeddings_helper(loader, store_name, path)

if __name__ == "__main__":
    # embedding_store = TextEmbeddingStore()
    embedding_store = PDFEmbeddingStore()
    embedding_store.store_embeddings(
        "embedding",
        "embeddings"
    )