import streamlit as st
import os
from langchain_openai import  ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from helper_fn import pdf_to_embeddings,format_doc,format_prompt

#Main RAG Chain Logic

def get_rag_chain(vectorstore, openai_api_key):
    """Creates the main RAG chain for querying."""
    llm = ChatOpenAI(model='gpt-4.1-nano', openai_api_key=openai_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # This is the RAG chain that takes a query and returns a response
    chain = {
        "context": retriever | RunnableLambda(format_doc),
        "query": RunnablePassthrough(),
    } | RunnableLambda(lambda x: format_prompt(x["context"], x["query"])) | llm

    return chain


#Streamlit UI

st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG Assistant")

# Sidebar for API key and file upload
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get your key from https://platform.openai.com/api-keys"
    )

    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF file here and click 'Process'", type="pdf")

# Main panel
if uploaded_file:
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_path = os.path.join(temp_dir, uploaded_file.name)
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
            with st.spinner("Processing your request..."):
                try:
                    # Create vector store from the PDF
                    vectorstore = pdf_to_embeddings(temp_path, openai_api_key)
                    
                    # Create and run the RAG chain
                    rag_chain = get_rag_chain(vectorstore, openai_api_key)
                    response = rag_chain.invoke(user_query)
                    
                    # Display the response
                    st.success("Here is the answer:")
                    st.write(response.content)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a PDF file using the sidebar to get started.")