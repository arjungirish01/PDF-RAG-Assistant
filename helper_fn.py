from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def pdf_to_embeddings(pdf_path, openai_api_key):
    """Loads a PDF, splits it into chunks, and creates a FAISS vector store."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    text_chunks = splitter.split_documents(docs)
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(text_chunks, embedding_model)


def format_doc(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def format_prompt(context: str, query: str):
    template = """
    You are an AI Assistant. Reply to the query: {query} using the context: {context} in natural language.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format_prompt(context=context, query=query).to_messages()
