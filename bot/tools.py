import os
import json
import logging
import yaml
from dotenv import load_dotenv

# LangChain & RAG imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from ragatouille import RAGPretrainedModel
import cohere

# Load environment variables and configuration
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Import prompt templates from your prompt module
from .prompt import error_prompt, result_rephraser_prompt

##########################################################
#                  Error & Rephrase Chains
##########################################################

class Chains:
    """
    Provides LangChain-based error and rephrasing chains using ChatGroq.
    """
    llm = ChatGroq(model=config["error-model-name"], groq_api_key=config["groq_api_key"])
    error_generator_chain = error_prompt | llm | StrOutputParser()
    result_rephraser_chain = result_rephraser_prompt | llm | StrOutputParser()

##########################################################
#              Existing Tools: Summaries & Code
##########################################################

@tool("summarize_text", description="Summarizes the provided text concisely.")
def summarize_text(text: str) -> str:
    """
    Returns a concise summary of the given text.
    """
    try:
        prompt = PromptTemplate(
            template="Summarize the following text succinctly:\n\n{text}",
            input_variables=["text"]
        )
        llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0.5,
            openai_api_key=config.get("openai_api_key")
        )
        return llm(prompt.format(text=text))
    except Exception as e:
        return Chains.error_generator_chain.invoke(input={"input": f"Error summarizing text: {e}"})

@tool("improve_code", description="Refactors and improves the given code snippet.")
def improve_code(code: str) -> str:
    """
    Returns an improved/refactored version of the provided code snippet along with explanations.
    """
    try:
        prompt = PromptTemplate(
            template="Improve the following code snippet, refactor it if needed, and explain your changes:\n\n{code}",
            input_variables=["code"]
        )
        llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0.3,
            openai_api_key=config.get("openai_api_key")
        )
        return llm(prompt.format(code=code))
    except Exception as e:
        return Chains.error_generator_chain.invoke(input={"input": f"Error improving code: {e}"})

@tool("ingest_document", description="Loads a document (PDF or text) and returns a summary of its content.")
def ingest_document(file_path: str) -> str:
    """
    Loads a document from the given file path and returns a summary of its content.
    If the file is a PDF, it uses PyPDFLoader; otherwise, it uses TextLoader.
    """
    try:
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        if splits:
            # For demonstration, summarize the first chunk
            summary = summarize_text(splits[0].page_content)
            return summary
        else:
            return "No content found in the document."
    except Exception as e:
        return Chains.error_generator_chain.invoke(input={"input": f"Error ingesting document: {e}"})

##########################################################
#      RAG Pipeline: EnhancedRetriever & rag_search
##########################################################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "jinaai/jina-colbert-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", config.get("cohere_api_key", "zB0M7xA8iUUCVBEBsf8AnN0dCc7jWW6Xga8ev15O"))
COHERE_MODEL = os.getenv("COHERE_MODEL", "rerank-english-v3.0")

class EnhancedRetriever:
    """
    Loads documents (PDF or text), creates a vector store with Chroma, and 
    indexes them with a RAG model. Then uses an ensemble retriever to get 
    relevant chunks. Optionally re-ranks results with Cohere.
    """
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
            if self.file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(self.file_path)
            else:
                loader = TextLoader(self.file_path, encoding='utf-8')
            self.documents = loader.load()
            self.splits = self.text_splitter.split_documents(self.documents)
            logger.info("Documents loaded and split successfully.")
        except Exception as e:
            logger.error("Error loading documents: %s", e)
            raise

    def create_vectorstores(self):
        try:
            self.chroma_db = Chroma.from_documents(
                self.splits,
                self.embeddings,
                persist_directory="./chroma_db"
            )
            collection_with_metadata = [
                f"{doc.page_content}\nMETADATA: {json.dumps(doc.metadata)}"
                for doc in self.splits
            ]
            self.rag.index(
                collection=collection_with_metadata,
                index_name="selise_index",
                max_document_length=600,
                split_documents=True
            )
            logger.info("Vector stores created successfully.")
        except Exception as e:
            logger.error("Error creating vector stores: %s", e)
            raise

    def create_ensemble_retriever(self):
        try:
            chroma_retriever = self.chroma_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10}
            )
            rag_retriever = self.rag.as_langchain_retriever(
                index_name="selise_index",
                k=12
            )
            self.retriever = EnsembleRetriever(
                retrievers=[chroma_retriever, rag_retriever],
                weights=[0.7, 0.3]
            )
            logger.info("Ensemble retriever created successfully.")
        except Exception as e:
            logger.error("Error creating ensemble retriever: %s", e)
            raise

    def rerank_results(self, documents, query):
        """
        Optional step: use Cohere to re-rank top results from the ensemble retriever.
        """
        try:
            rerank_request = [doc.page_content for doc in documents]
            response = self.cohere_client.rerank(
                model=COHERE_MODEL,
                query=query,
                documents=rerank_request,
                top_n=len(documents)
            )
            ranked_docs = [documents[result.index] for result in response.results]
            logger.info("Documents reranked successfully.")
            return ranked_docs
        except Exception as e:
            logger.error("Error in reranking: %s", e)
            return documents

@tool("rag_search", description="Performs RAG-based search on loaded documents, returning a top snippet.")
def rag_search(query: str, file_path: str) -> str:
    """
    A tool that uses the EnhancedRetriever to load a file, retrieve relevant docs,
    optionally re-rank them, and return a snippet of the top result.
    """
    try:
        retriever_system = EnhancedRetriever(file_path)
        docs = retriever_system.retriever.get_relevant_documents(query)
        reranked_docs = retriever_system.rerank_results(docs, query)
        if not reranked_docs:
            return "No documents found for your query."
        top_doc = reranked_docs[0].page_content
        snippet = top_doc[:500] + ("..." if len(top_doc) > 500 else "")
        return f"Top Document Snippet:\n{snippet}"
    except Exception as e:
        return Chains.error_generator_chain.invoke(input={"input": f"Error performing RAG search: {e}"})