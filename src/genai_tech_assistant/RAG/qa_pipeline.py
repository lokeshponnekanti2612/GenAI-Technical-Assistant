from dataclasses import dataclass
from typing import List

from genai_tech_assistant.Embeddings.embedding_client import EmbeddingClient
from genai_tech_assistant.VectorStore.chroma_store import ChromaVectorStore, RetrievedChunk
from genai_tech_assistant.LLM.ollama_client import OllamaLLMClient
from genai_tech_assistant.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    retrieved: List[RetrievedChunk]

# adding retreived chunks and building the input context


def build_context(chunks: List[RetrievedChunk], max_chars: int = 4000) -> str:

    if not chunks:
        return ""

    parts: List[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "unknown")

        block = (
            f"[Chunk {i} | source={source} | page={page} | distance={chunk.distance:.4f}]\n"
            f"{chunk.text.strip()}\n"
        )

        if total_chars + len(block) > max_chars:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts).strip()


def call_llm(question: str, context: str) -> str:

    if not context.strip():
        return "I don't know based on the retrieved context."

    try:
        client = OllamaLLMClient()
        return client.generate_answer(question, context)
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"LLM call failed: {e}"


def answer_question(question: str, top_k: int = 3, embedder: EmbeddingClient | None = None, store: ChromaVectorStore | None = None,) -> RAGAnswer:

    if not question or not question.strip():
        return RAGAnswer(
            question=question,
            answer="Question is empty.",
            retrieved=[],
        )

    if embedder is None:
        embedder = EmbeddingClient()

    if store is None:
        store = ChromaVectorStore()

    logger.info(f"Answering question with RAG: {question}")

    # embedding the user question
    query_vec = embedder.embed_text(question)

    if not query_vec:
        return RAGAnswer(
            question=question,
            answer="Failed to create an embedding for the question.",
            retrieved=[],
        )

    # bringing out relevant embedding from chromadb
    retrieved = store.query(query_vec, top_k=top_k)

    # convering retreived chunks to context
    context = build_context(retrieved)

    # final call
    answer = call_llm(question, context)

    return RAGAnswer(
        question=question,
        answer=answer,
        retrieved=retrieved,
    )
