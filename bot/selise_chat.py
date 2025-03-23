
# bot/selise_chat.py
import os, json, logging, cohere
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from ragatouille import RAGPretrainedModel

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "jinaai/jina-colbert-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your_cohere_api_key")
COHERE_MODEL = os.getenv("COHERE_MODEL", "rerank-english-v3.0")

class EnhancedRetriever:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.rag = RAGPretrainedModel.from_pretrained(RAG_MODEL_NAME)
        self.cohere_client = cohere.Client(COHERE_API_KEY)
        self.load_documents()
        self.create_vectorstores()
        self.create_ensemble_retriever()

    def load_documents(self):
        try:
            loader = PyPDFLoader(self.file_path) if self.file_path.endswith('.pdf') else TextLoader(self.file_path, encoding='utf-8')
            self.documents = loader.load()
            self.splits = self.text_splitter.split_documents(self.documents)
            logger.info("Documents loaded and split.")
        except Exception as e:
            logger.error("Error loading documents: %s", e)
            raise

    def create_vectorstores(self):
        try:
            self.chroma_db = Chroma.from_documents(self.splits, self.embeddings, persist_directory="./chroma_db")
            collection_with_metadata = [f"{doc.page_content}\nMETADATA: {json.dumps(doc.metadata)}" for doc in self.splits]
            self.rag.index(collection=collection_with_metadata, index_name="selise_index", max_document_length=600, split_documents=True)
            logger.info("Vector stores created.")
        except Exception as e:
            logger.error("Error creating vector stores: %s", e)
            raise

    def create_ensemble_retriever(self):
        try:
            chroma_retriever = self.chroma_db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
            rag_retriever = self.rag.as_langchain_retriever(index_name="selise_index", k=12)
            self.retriever = EnsembleRetriever(retrievers=[chroma_retriever, rag_retriever], weights=[0.7, 0.3])
            logger.info("Ensemble retriever created.")
        except Exception as e:
            logger.error("Error creating ensemble retriever: %s", e)
            raise

    def rerank_results(self, documents, query):
        try:
            rerank_request = [doc.page_content for doc in documents]
            response = self.cohere_client.rerank(model=COHERE_MODEL, query=query, documents=rerank_request, top_n=len(documents))
            ranked_docs = [documents[result.index] for result in response.results]
            logger.info("Documents reranked.")
            return ranked_docs
        except Exception as e:
            logger.error("Error in reranking: %s", e)
            return documents

def create_qa_chain(retriever_system, groq_api_key=None):
    # Custom SELISE prompt template
    response_template = """[INST]
<>
You are the SELISE AI Assistant—a knowledgeable expert on SELISE Digital Platforms.
Provide clear, engaging responses that include details such as Name, Role, and Email for any contact information requested.

User Question:
{question}

Context:
{context}
</>

**SELISE AI:** 😊
"""
    prompt = PromptTemplate(template=response_template, input_variables=["context", "question"])
    groq_chat = ChatGroq(groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY", "your_groq_api_key"),
                          model_name=GROQ_MODEL)
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=groq_chat,
        chain_type="stuff",
        retriever=retriever_system.retriever,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
        return_source_documents=True
    )
    return qa_chain
