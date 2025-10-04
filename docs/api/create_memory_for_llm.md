# API: create_memory_for_llm

Builds the FAISS vector store from PDFs placed in `data/`.

Source: `create_memory_for_llm.py`

## Constants
- `DATA_PATH`: Directory containing input PDFs. Default: `data/`.
- `DB_FAISS_PATH`: Output directory for FAISS index. Default: `vectorstore/db_faiss`.

## Functions

### load_pdf_files(data: str) -> List[Document]
Loads PDF files from a directory using `DirectoryLoader` and `PyMuPDFLoader`.

- **Parameters**:
  - `data`: Directory path containing PDFs (e.g., `data/`)
- **Returns**: List of LangChain `Document` objects

Example:
```python
from create_memory_for_llm import load_pdf_files
pdf_docs = load_pdf_files("data/")
print(len(pdf_docs))
```

---

### create_chunks(extracted_data: List[Document]) -> List[Document]
Splits documents into overlapping chunks with `RecursiveCharacterTextSplitter`.

- **Parameters**:
  - `extracted_data`: Documents to split
- **Returns**: List of chunked `Document` objects
- **Defaults**:
  - `chunk_size=500`, `chunk_overlap=50`

Example:
```python
from create_memory_for_llm import create_chunks
chunks = create_chunks(pdf_docs)
```

---

### get_embedding_model() -> Embeddings
Creates the Hugging Face embeddings model `sentence-transformers/all-MiniLM-L6-v2`.

- **Parameters**: None
- **Returns**: Embeddings object compatible with LangChain

Example:
```python
from create_memory_for_llm import get_embedding_model
emb = get_embedding_model()
```

## Script usage (build and save FAISS)
Running the module as a script will:

1. Load PDFs from `DATA_PATH`
2. Create chunks
3. Create embeddings
4. Build FAISS index and save to `DB_FAISS_PATH`

```bash
python create_memory_for_llm.py
```

Programmatic equivalent:
```python
from create_memory_for_llm import load_pdf_files, create_chunks, get_embedding_model
from langchain_community.vectorstores import FAISS

pdf_docs = load_pdf_files("data/")
chunks = create_chunks(pdf_docs)
emb = get_embedding_model()
store = FAISS.from_documents(chunks, emb)
store.save_local("vectorstore/db_faiss")
```

## Notes
- For best results, ensure PDFs contain selectable text. Scanned PDFs may need OCR.
- You can change `DATA_PATH`, `DB_FAISS_PATH`, chunking parameters, or the embedding model to fit your use case.
