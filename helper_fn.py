import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def _get_faiss_dir(pdf_path: str) -> str:
    return pdf_path + "_faiss_index"

def pdf_to_embeddings(pdf_path: str, openai_api_key: str, use_cache: bool = True, use_openai: bool = True):
    """
    Loads a PDF, splits it into chunks, creates embeddings and a FAISS vector store.
    If use_cache is True and a cached index exists, it will be loaded instead of recomputing.
    """
    index_dir=_get_faiss_dir(pdf_path)

    # Try load cache
    if use_cache and os.path.exists(index_dir):
        try:
            embedding_model=(
                OpenAIEmbeddings(openai_api_key=openai_api_key) if use_openai else None
            )
            if embedding_model is None:
                raise ValueError("No embedding model specified for local loading")
            vectorstore=FAISS.load_local(index_dir,
                                         embeddings=embedding_model,
                                         allow_dangerous_deserialization=True
                                         )
        except Exception as e:
            print(f"[WARN] Failed to load cached FAISS index {e}")
            

    # Load and split
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    text_chunks = splitter.split_documents(docs)

    # Choose embedding backend
    if use_openai:
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        raise RuntimeError("Non-OpenAI embeddings not configured. Set use_openai=True or implement local backend.")

    # Build FAISS index
    vectorstore = FAISS.from_documents(text_chunks, embedding_model)

    # Save cache
    if use_cache:
        try:
            os.makedirs(index_dir,exist_ok=True)
            vectorstore.save_local(index_dir)
        except Exception as e:
            print(f"[WARN] failed to save FAISS Index {e}")

    return vectorstore


def format_doc(docs: List) -> str:
    """
    Turn a list of Document objects into a single context string.
    Optionally include page numbers or metadata if available.
    """
    pieces = []
    for d in docs:
        # include a small metadata header if present
        md = d.metadata or {}
        if "page" in md:
            pieces.append(f"[Page {md['page']}] {d.page_content}")
        else:
            pieces.append(d.page_content)
    return "\n\n".join(pieces)


def format_prompt(context: str, query: str) -> str:
    """
    A hardened prompt that instructs the LLM to answer ONLY using the provided context.
    If the context does not contain the answer, the model must state that explicitly.
    """
    template = (
        "You are an assistant that answers questions using ONLY the provided CONTEXT. "
        "Do NOT use external knowledge or invent facts. If the context does not contain "
        "enough information to answer, respond: \"I cannot find relevant information in the provided document.\". \n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer in clear, concise language and, when relevant, cite which page or snippet from the context you used."
    )
    return template.format(context=context, query=query)
