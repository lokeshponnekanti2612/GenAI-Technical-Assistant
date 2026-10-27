# GenAI Technical Assistant

A local Retrieval-Augmented Generation (RAG) assistant for answering questions over technical manuals — fully local, no paid API required.

## How it works

1. **Ingestion** (`Ingestion/pdf_ingestor.py`) — parses PDFs with `pypdf`, splits text into overlapping sentence-based chunks.
2. **Embeddings** (`Embeddings/embedding_client.py`) — embeds chunks locally with a Sentence-Transformers model (`all-MiniLM-L6-v2`), no external API call.
3. **Vector store** (`VectorStore/chroma_store.py`) — persists embeddings in a local Chroma collection.
4. **Retrieval + generation** (`RAG/qa_pipeline.py`) — retrieves the top-k relevant chunks for a question and passes them as context to a local LLM.
5. **LLM** (`LLM/ollama_client.py`) — generates the answer via a local Ollama model (`llama3.2`), with streaming support.

## Interfaces

- `chat_cli.py` — terminal chat loop; prints the answer plus retrieved source chunks (file + page number) for traceability.
- `web_app.py` — FastAPI web app serving a chat UI (`web/index.html`) with streaming responses.

## Data

`data/raw_findings/` holds the source PDFs the assistant answers questions about (a diagnostic/troubleshooting manual, an operating manual, and a technical guide).

## Stack

Python, FastAPI, ChromaDB, Sentence-Transformers, Ollama, pypdf.
