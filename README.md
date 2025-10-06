# Aegis 🤖: A Q&A Chatbot for Medical Information

This project is a medical Q&A chatbot named **Aegis**, built using LangChain, Hugging Face, and Groq. It's designed to answer questions based on a provided set of PDF documents. The chatbot uses a Retrieval-Augmented Generation (RAG) architecture to find relevant information in the documents before generating a response.

## 🌐 Live Demo

**Try Aegis now:** [https://aegis--the-ai-medical-chatbot-l2mrvkzmomehccz2hvssv8.streamlit.app/](https://aegis--the-ai-medical-chatbot-l2mrvkzmomehccz2hvssv8.streamlit.app/)

## ✨ Features

-   **Retrieval-Augmented Generation (RAG):** Aegis retrieves information from a knowledge base (PDFs) to provide accurate answers.
-   **Fast and Efficient:** Utilizes the high-performance Groq API for rapid inference.
-   **Persistent Memory:** Uses FAISS for an efficient vector store to remember and retrieve information from the documents.
-   **Streamlit Interface:** Provides a user-friendly web interface for interaction.

## ⚙️ How It Works

1.  **Load and Process Documents:** PDF files from the `data/` directory are loaded and split into smaller chunks.
2.  **Create Embeddings:** The `sentence-transformers/all-MiniLM-L6-v2` model is used to convert the text chunks into numerical vectors (embeddings).
3.  **Vector Store:** These embeddings are stored in a FAISS vector database (`vectorstore/db_faiss`). This acts as the chatbot's long-term memory.
4.  **User Query:** When a user asks a question, the query is also converted into an embedding.
5.  **Retrieval:** Aegis retrieves the most semantically similar text chunks from the FAISS database.
6.  **Generation:** These retrieved chunks, along with the user's question, are fed to a large language model (LLM) hosted by Groq to generate a final, contextually relevant answer.

## 🚀 Setup and Installation

### Prerequisites

-   Python 3.8+
-   A Groq API Key and a Hugging Face API token.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
