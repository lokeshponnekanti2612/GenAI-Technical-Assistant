from dataclasses import dataclass
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from genai_tech_assistant.logging_config import get_logger
from genai_tech_assistant.Ingestion.pdf_ingestor2 import Documentchunk

logger = get_logger(__name__)


@dataclass
class EmbeddedChunk:
    id: str
    embedding: list[float]
    text: str
    metadata: Dict[str, Any]


class EmbeddingClient:
    def __init__(self, model_name: str | None = None) -> None:
        if model_name is None:
            model_name = "all-MiniLM-L6-v2"

        logger.info(f"Loading local embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)

    # used to embed texts offline the documents
    def embed_texts(self, texts: List[str]) -> List[list[float]]:
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts with local model")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=False,
            show_progress_bar=False,
        )

        return [emb.tolist() for emb in embeddings]
    # used to clean the user question and then embed it by passing it to embed_texts

    def embed_text(self, text: str) -> list[float]:
        stripped = text.strip()
        if not stripped:
            return []

        return self.embed_texts([stripped])[0]


def embed_chunks(chunks: List[Documentchunk], embedder: EmbeddingClient, batch_size: int = 64,) -> List[EmbeddedChunk]:
    if not chunks:
        logger.warning(
            "embed_chunks called with no chunks; returning empty list.")
        return []

    logger.debug(f"Embedding {len(chunks)} chunks (batch_size={batch_size})")
    embedded: List[EmbeddedChunk] = []

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch = chunks[start:end]

        texts = [c.text for c in batch]
        ids = [c.id for c in batch]
        metadatas = [c.metadata for c in batch]

        vectors = embedder.embed_texts(texts)

        if len(vectors) != len(batch):
            logger.error(
                f"Embedding count ({len(vectors)}) != batch size ({len(batch)}) "
                f"for slice {start}:{end}"
            )
            continue

        logger.info(
            f"Embedded batch {start}–{end} (size {len(batch)}) into vectors "
            f"of dim {len(vectors[0]) if vectors else 'unknown'}"
        )

        for chunk, vec in zip(batch, vectors):
            embedded.append(
                EmbeddedChunk(
                    id=chunk.id,
                    embedding=vec,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )

    logger.info(f"Total embedded chunks created: {len(embedded)}")
    return embedded


if __name__ == "__main__":
    from genai_tech_assistant.Ingestion.pdf_ingestor2 import ingest_pdfs

    chunks = ingest_pdfs()
    print(f"Total chunks loaded: {len(chunks)}")

    embedder = EmbeddingClient()
    embedded_chunks = embed_chunks(chunks, embedder)

    print(f"Total embedded chunks: {len(embedded_chunks)}")

    for chunk in embedded_chunks[:3]:
        print("=" * 80)
        print("ID:", chunk.id)
        print("Metadata:", chunk.metadata)
        print("Vector dim:", len(chunk.embedding))
        print("First 10 values:", chunk.embedding[:10])
        print("Text:", chunk.text[:200])
