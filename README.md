# PDF RAG Assistant

A user-friendly web application that leverages a Retrieval-Augmented Generation (RAG) pipeline to answer questions about the content of any uploaded PDF document.

**

<img width="1853" height="581" alt="RAG-APP" src="https://github.com/user-attachments/assets/9127a30a-7fa9-4290-ada1-bfb709c6d682" />

---

## About The Project

This project is a powerful, self-contained AI assistant that allows users to have a conversation with their documents. It uses the RAG (Retrieval-Augmented Generation) architecture to provide accurate, context-aware answers based solely on the information within a provided PDF file.

The application is built with a modern tech stack, featuring a Streamlit front-end for a seamless user experience, a LangChain back-end to orchestrate the AI pipeline, and Docker for easy, reproducible deployment. It's designed not only as a useful tool but also as a demonstration of building and deploying end-to-end Generative AI applications.

---

## Tech Stack

* **Python:** The core programming language.
* **LangChain:** Framework for orchestrating the RAG pipeline and interacting with the LLM.
* **OpenAI:** For state-of-the-art language models (GPT-4) and text embeddings.
* **FAISS:** For efficient similarity search in the vector store.
* **Streamlit:** To create the interactive and user-friendly web interface.
* **Docker:** For containerizing the application for consistent, cross-platform deployment.

---

## Features

* **Dynamic PDF Upload:** Users can upload any PDF document through the web interface.
* **Contextual Q&A:** The AI answers questions using only the information present in the uploaded document.
* **Secure API Key Handling:** Accepts OpenAI API keys via a secure input field or environment variables.
* **Containerized & Reproducible:** The included `Dockerfile` ensures the application can be run anywhere with Docker installed.

---

## Getting Started

You can run this application either locally on your machine or using Docker.

### Prerequisites

* Python 3.9+
* Docker (for the Docker setup)
* An OpenAI API Key

### 1. Local Setup

**1. Clone the repository:**
```bash
git clone [(https://github.com/arjungirish01/PDF-RAG-Assistant.git)]
cd PDF-RAG-Assistant
```

***2. Create and activate a virtual environment:***
```bash
# Create the environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
.\venv\Scripts\activate
```

***3. Install dependencies:***
```bash
pip install requirements.txt
```

***4. Set your OpenAI API Key:***
```bash
# macOS/Linux
export OPENAI_API_KEY="your_api_key_here"

# Windows
set OPENAI_API_KEY="your_api_key_here"
```
**(Alternatively, you can paste the key directly into the app's sidebar.)**

***5. Run the application:***
```bash
streamlit run app.py
```

***2. Docker Setup**
**1. Build the Docker image:**
```bash
docker build -t rag-app .
```

**2. Run the Docker container:**
```bash
docker run -p 8501:8501 -e OPENAI_API_KEY="your_api_key_here" rag-app
```
