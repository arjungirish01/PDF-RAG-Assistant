import streamlit as st
import os
import time
from dotenv import load_dotenv
from helper_fn import pdf_to_embeddings, format_doc, format_prompt
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableMap
load_dotenv()

# Streamlit Configuration
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG ASSISTANT ")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get your key from https://platform.openai.com/api-keys"
    )

    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF file and click 'Process'", type="pdf")

    st.markdown("---")
    use_cache = st.checkbox("Use persistent FAISS cache (recommended)", value=True)
    show_context = st.checkbox("Show retrieved context", value=False)

# Helper to handle LLM output
def _response_to_text(resp):
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)

# Main app logic
if uploaded_file:
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    # Save uploaded PDF to disk
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.info("PDF uploaded. You can now ask questions about its content.")

    user_query = st.text_input("Ask a question about your PDF:")

    if st.button("Get Answer"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        elif not user_query:
            st.warning("Please enter a question.")
        else:
            try:
                start_all = time.time()
                st.write("Building or loading FAISS index...")

                # Load or build FAISS vector store
                vectorstore = pdf_to_embeddings(
                    pdf_path=temp_path,
                    openai_api_key=openai_api_key,
                    use_cache=use_cache
                )

                # Determine if cache was reused or built fresh
                faiss_dir = temp_path + "_faiss_index"
                if os.path.exists(faiss_dir):
                    st.success(f"Using persistent FAISS index at `{faiss_dir}`")
                else:
                    st.info("Built new FAISS index from PDF text chunks.")

            
                # Create modular RAG pipeline

                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                context_runnable = retriever | RunnableLambda(format_doc)
                prompt_runnable = RunnableLambda(lambda x: format_prompt(x["context"], x["query"]))

                llm = ChatOpenAI(
                    model="gpt-4.1-nano",
                    openai_api_key=openai_api_key,
                    temperature=0.2,
                )

                rag_pipeline = (
                    RunnableMap({
                        "context": context_runnable,
                        "query": RunnablePassthrough(),
                    })
                    | prompt_runnable
                    | llm
                )

               
                # Retrieve top documents
            
                top_docs = retriever.invoke(user_query)
                if not top_docs:
                    st.warning("No relevant context found in the document.")
                    st.write("Try rephrasing your question or uploading another document.")
                else:
                    if show_context:
                        st.markdown("Retrieved Context Chunks")
                        for i, d in enumerate(top_docs[:4], 1):
                            md = d.metadata or {}
                            page_info = f"(page {md.get('page', '?')})" if "page" in md else ""
                            st.write(f"Chunk {i} {page_info}:")
                            st.write(d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""))

                    
                    # Run RAG pipeline
                
                    t0 = time.time()
                    resp = rag_pipeline.invoke(user_query)
                    latency = time.time() - t0

                    answer = _response_to_text(resp)
                    st.success("Answer:")
                    st.write(answer)

                    total_time = time.time() - start_all
                    st.markdown(
                        f"Query time: {latency:.2f}s (generation) | Total: {total_time:.2f}s"
                    )

            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")

else:
    st.info("Please upload a PDF using the sidebar to get started.")
