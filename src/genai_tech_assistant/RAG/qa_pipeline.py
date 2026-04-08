import re
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


@dataclass
class RAGRetrieval:
    question: str
    retrieved: List[RetrievedChunk]
    context: str
    response_override: str | None = None


GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|hiya|good morning|good afternoon|good evening|thanks|thank you)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def get_direct_response_for_non_rag_query(question: str) -> str | None:
    stripped = question.strip()
    if not stripped:
        return "Question is empty."

    if GREETING_PATTERN.fullmatch(stripped):
        return (
            "Hi! I can help with questions about the technical manuals. "
            "Try asking about a fault code, component, procedure, or safety step."
        )

    return None


def is_low_confidence_retrieval(chunks: List[RetrievedChunk]) -> bool:
    if not chunks:
        return True

    best_distance = chunks[0].distance
    if best_distance is None:
        return True

    # Chroma distances vary by embedding/metric setup. This threshold is a
    # conservative starting point to reject very weak semantic matches.
    return best_distance > 1.2

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


def retrieve_context_for_question(
    question: str,
    top_k: int = 3,
    embedder: EmbeddingClient | None = None,
    store: ChromaVectorStore | None = None,
) -> RAGRetrieval:
    direct_response = get_direct_response_for_non_rag_query(question)
    if direct_response is not None:
        return RAGRetrieval(
            question=question,
            retrieved=[],
            context="",
            response_override=direct_response,
        )

    if embedder is None:
        embedder = EmbeddingClient()

    if store is None:
        store = ChromaVectorStore()

    logger.info(f"Answering question with RAG: {question}")

    # embedding the user question
    query_vec = embedder.embed_text(question)

    if not query_vec:
        return RAGRetrieval(
            question=question,
            retrieved=[],
            context="",
        )

    # bringing out relevant embedding from chromadb
    retrieved = store.query(query_vec, top_k=top_k)

    if is_low_confidence_retrieval(retrieved):
        logger.info(
            "Rejecting weak retrieval for question '%s' (best distance=%s)",
            question,
            retrieved[0].distance if retrieved else "n/a",
        )
        return RAGRetrieval(
            question=question,
            retrieved=[],
            context="",
            response_override="The answer is not explicitly found in the retrieved context.",
        )

    # convering retreived chunks to context
    context = build_context(retrieved)

    return RAGRetrieval(
        question=question,
        retrieved=retrieved,
        context=context,
        response_override=None,
    )


def answer_question(question: str, top_k: int = 3, embedder: EmbeddingClient | None = None, store: ChromaVectorStore | None = None,) -> RAGAnswer:
    retrieval = retrieve_context_for_question(
        question,
        top_k=top_k,
        embedder=embedder,
        store=store,
    )

    if retrieval.response_override is not None:
        return RAGAnswer(
            question=question,
            answer=retrieval.response_override,
            retrieved=retrieval.retrieved,
        )

    if not retrieval.context:
        return RAGAnswer(
            question=question,
            answer="The answer is not explicitly found in the retrieved context.",
            retrieved=retrieval.retrieved,
        )

    # final call
    answer = call_llm(question, retrieval.context)

    return RAGAnswer(
        question=retrieval.question,
        answer=answer,
        retrieved=retrieval.retrieved,
    )
