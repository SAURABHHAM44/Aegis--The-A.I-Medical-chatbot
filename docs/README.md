# Aegis Documentation

Comprehensive documentation for all public APIs, functions, and components in this repository. This project provides a Retrieval-Augmented Generation (RAG) chatbot (Streamlit UI) backed by a FAISS vector store created from local PDFs.

- **Source code**: `medibot.py`, `create_memory_for_llm.py`
- **Docs**: `docs/README.md` (this file), `docs/api/medibot.md`, `docs/api/create_memory_for_llm.md`

## Quick start

### 1) Prepare environment
Create a `.env` file with required keys:

```bash
GROQ_API_KEY=your_groq_api_key
# Optional if you want to use Hugging Face Inference API via load_llm()
HF_TOKEN=your_huggingface_token
```

Install dependencies:

```bash
pip install -U pip
pip install streamlit langchain langchain-community langchain-core langchain-huggingface langchain-groq faiss-cpu pypdf pymupdf python-dotenv
```

### 2) Put PDFs in `data/`
Place your source PDF files in the `data/` directory.

### 3) Build the vector store
This creates `vectorstore/db_faiss/` with the document embeddings.

```bash
python create_memory_for_llm.py
```

### 4) Run the chatbot UI

```bash
streamlit run medibot.py
```

Open the local URL that Streamlit prints (usually `http://localhost:8501`).

## Architecture overview

- **Data ingestion and indexing** (`create_memory_for_llm.py`)
  - Loads PDFs from `data/`
  - Splits documents into overlapping chunks
  - Embeds with `sentence-transformers/all-MiniLM-L6-v2`
  - Saves into FAISS at `vectorstore/db_faiss`

- **Question answering UI** (`medibot.py`)
  - Streamlit chat interface with session state
  - Retrieves top-k chunks (default k=3) from FAISS
  - Uses Groq-hosted LLM for generation
  - Customizable prompt via a `PromptTemplate`

## Configuration

- **Environment variables**
  - **GROQ_API_KEY**: Required to use the Groq-hosted chat model.
  - **HF_TOKEN**: Optional. Required only when using `load_llm()` with the Hugging Face Inference API.

- **Paths**
  - **Data**: `data/`
  - **Vector store**: `vectorstore/db_faiss`

- **Models**
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
  - **Chat (default)**: `meta-llama/llama-4-maverick-17b-128e-instruct` via Groq

## Usage examples

### Build vector store programmatically
```python
from create_memory_for_llm import load_pdf_files, create_chunks, get_embedding_model
from langchain_community.vectorstores import FAISS

# 1) Load and chunk
documents = load_pdf_files("data/")
chunks = create_chunks(documents)

# 2) Embed and persist
embeddings = get_embedding_model()
faiss_store = FAISS.from_documents(chunks, embeddings)
faiss_store.save_local("vectorstore/db_faiss")
```

### Customize the prompt in the UI
```python
from medibot import set_custom_prompt
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

CUSTOM_PROMPT = """
Use the context to answer succinctly.
If unsure, say you don't know.
Context: {context}
Question: {question}
"""

prompt = set_custom_prompt(CUSTOM_PROMPT)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.0,
    ),
    chain_type="stuff",
    retriever=faiss_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
```

### Use Hugging Face Inference API instead of Groq
```python
from medibot import load_llm, set_custom_prompt, get_vectorstore
from langchain.chains import RetrievalQA

llm = load_llm(
    huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    HF_TOKEN="<your_hf_token>"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=get_vectorstore().as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt("Context: {context}\nQuestion: {question}")},
)
```

## API Reference

- [`docs/api/medibot.md`](api/medibot.md)
- [`docs/api/create_memory_for_llm.md`](api/create_memory_for_llm.md)

## Troubleshooting

- **No such file or directory: vectorstore/db_faiss**: Build the vector store first with `python create_memory_for_llm.py`.
- **Authentication errors**: Ensure `GROQ_API_KEY` is set. For Hugging Face Inference API, set `HF_TOKEN`.
- **Empty or irrelevant answers**: Check your `data/` PDFs, increase `k` in the retriever, or refine the prompt.
