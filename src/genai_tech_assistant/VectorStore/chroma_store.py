# src/genai_tech_assistant/VectorStore/chroma_store.py

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb

from genai_tech_assistant.config import settings
from genai_tech_assistant.logging_config import get_logger
from genai_tech_assistant.Embeddings.embedding_client import EmbeddedChunk

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:

    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float


class ChromaVectorStore:

    def __init__(self, collection_name: str = "technician_docs", persist_dir: Path | None = None, ) -> None:
        # where to store Chroma files on disk
        if persist_dir is None:
            persist_dir = settings.vector_store_dir  # /data/vector_store

        self.persist_dir = Path(persist_dir)  # converts to path object
        self.collection_name = collection_name  # for now its technician_docs

        # intializes at data/vector_store/
        logger.info(f"Initializing Chroma at: {self.persist_dir}")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir))  # connecting to chromadb
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name)
        logger.info(f"Using Chroma collection: {self.collection_name}")

    def index_embedded_chunks(self, embedded_chunks: List[EmbeddedChunk], batch_size: int = 128) -> None:

        if not embedded_chunks:
            logger.warning(
                "No embedded chunks provided to index so nothing to do.")
            return

        logger.info(
            f"Indexing {len(embedded_chunks)} embedded chunks into Chroma "
            f"(batch_size={batch_size})"
        )

        for start in range(0, len(embedded_chunks), batch_size):
            end = start + batch_size
            batch = embedded_chunks[start:end]

            ids = [c.id for c in batch]
            embeddings = [c.embedding for c in batch]
            documents = [c.text for c in batch]
            metadata = [c.metadata for c in batch]

            logger.info(
                f"Adding batch {start}–{min(end, len(embedded_chunks))} "
                f"(size {len(batch)}) to collection '{self.collection_name}'"
            )

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
            )

        logger.info("Finished indexing embedded chunks into Chroma.")

    # to retrieve relevant chunks
    def query(self, query_embedding: list[float],  top_k: int = 5,) -> List[RetrievedChunk]:

        if not query_embedding:
            logger.warning(
                "Empty query_embedding passed to query(); returning [].")
            return []

        logger.info(
            f"Querying Chroma collection '{self.collection_name}' for top_k={top_k}"
        )

        # takes embedded vector of user query
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # the .query() returns 2D lists where one list per query

        ids_lists = res.get("ids", [[]])
        docs_lists = res.get("documents", [[]])
        metas_lists = res.get("metadatas", [[]])
        dists_lists = res.get("distances", [[]])

        if not ids_lists or not ids_lists[0]:
            logger.info("Chroma query returned no results.")
            return []
         # so we look at index 0 because we are doing 1 query per time
        ids = ids_lists[0]
        docs = docs_lists[0]
        metas = metas_lists[0]
        dists = dists_lists[0]

        retrieved: List[RetrievedChunk] = []

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            retrieved.append(
                RetrievedChunk(
                    id=cid,
                    text=doc,
                    metadata=meta or {},
                    distance=dist,
                )
            )

        logger.info(
            f"Chroma query returned {len(retrieved)} results "
            f"(best distance={retrieved[0].distance if retrieved else 'n/a'})"
        )

        return retrieved


if __name__ == "__main__":
    from genai_tech_assistant.Ingestion.pdf_ingestor2 import ingest_pdfs
    from genai_tech_assistant.Embeddings.embedding_client import EmbeddingClient, embed_chunks

    chunks = ingest_pdfs()
    print(f"Total chunks loaded: {len(chunks)}")

    embedder = EmbeddingClient()
    embedded_chunks = embed_chunks(chunks, embedder)
    print(f"Total embedded chunks: {len(embedded_chunks)}")

    store = ChromaVectorStore()
    store.index_embedded_chunks(embedded_chunks)

    print(f"Collection name: {store.collection_name}")
    print(f"Vector count in collection: {store._collection.count()}")

    # simple test query
    test_question = "What are the three requirements for engine combustion?"
    query_vec = embedder.embed_text(test_question)
    results = store.query(query_vec, top_k=3)

    print("\nTest query:", test_question)
    for i, r in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result #{i}")
        print("Distance:", r.distance)
        print("Metadata:", r.metadata)
        print("Text:", r.text[:300])
