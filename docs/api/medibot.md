# API: medibot

Streamlit-based chat UI that performs retrieval-augmented generation (RAG) over a FAISS vector store.

Source: `medibot.py`

## Environment
- `GROQ_API_KEY` (required): API key for Groq-hosted models.
- `HF_TOKEN` (optional): Token for Hugging Face Inference API, used only with `load_llm()`.

## Constants
- `DB_FAISS_PATH`: Path to the FAISS index directory. Default: `vectorstore/db_faiss`.

## Functions

### get_vectorstore() -> FAISS
Loads the FAISS vector store from `DB_FAISS_PATH`. Cached across reruns via Streamlit's `@st.cache_resource`.

- **Parameters**: None
- **Returns**: `FAISS` vector store instance
- **Raises**: Exceptions from `FAISS.load_local` if the path is missing or corrupted

Example:
```python
from medibot import get_vectorstore
store = get_vectorstore()
docs = store.similarity_search("hypertension treatment", k=3)
```

---

### set_custom_prompt(custom_prompt_template: str) -> PromptTemplate
Constructs a LangChain `PromptTemplate` using the provided template. The template must define `{context}` and `{question}` placeholders.

- **Parameters**:
  - `custom_prompt_template`: Prompt text containing `{context}` and `{question}`
- **Returns**: `PromptTemplate`

Example:
```python
from medibot import set_custom_prompt
prompt = set_custom_prompt("Context: {context}\nQuestion: {question}")
```

---

### load_llm(huggingface_repo_id: str, HF_TOKEN: str) -> BaseLanguageModel
Creates a Hugging Face Inference API LLM client via `HuggingFaceEndpoint`. Primarily useful if you prefer HF Inference instead of Groq. Not used by `main()`.

- **Parameters**:
  - `huggingface_repo_id`: Model repo id on Hugging Face, e.g. `mistralai/Mistral-7B-Instruct-v0.3`
  - `HF_TOKEN`: Personal access token for Hugging Face
- **Returns**: LLM instance compatible with LangChain

Example:
```python
from medibot import load_llm
llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3", HF_TOKEN="<your_token>")
```

---

### main() -> None
Launches the Streamlit chat UI.

- **Behavior**:
  - Renders a chat interface
  - Retrieves context from FAISS using `get_vectorstore()`
  - Builds a `RetrievalQA` chain
  - Uses Groq's `meta-llama/llama-4-maverick-17b-128e-instruct` by default
  - Displays the model's answer and the retrieved source documents

Run:
```bash
streamlit run medibot.py
```

## End-to-end example

```python
import os
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from medibot import get_vectorstore, set_custom_prompt

os.environ["GROQ_API_KEY"] = "<your_groq_key>"

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = set_custom_prompt(
    """
Use the context to answer succinctly.
If unsure, say you don't know.
Context: {context}
Question: {question}
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.0,
        groq_api_key=os.environ["GROQ_API_KEY"],
    ),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

result = qa_chain.invoke({"query": "What are first-line treatments for type 2 diabetes?"})
print(result["result"])  # model answer
print(result["source_documents"])  # retrieved chunks
```
