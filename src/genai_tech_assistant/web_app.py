import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from genai_tech_assistant.LLM.ollama_client import OllamaLLMClient
from genai_tech_assistant.RAG.qa_pipeline import answer_question, retrieve_context_for_question
from genai_tech_assistant.logging_config import get_logger

logger = get_logger(__name__)

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_DIR / "web"
STATIC_DIR = FRONTEND_DIR / "static"
INDEX_FILE = FRONTEND_DIR / "index.html"

app = FastAPI(
    title="GenAI Technical Assistant",
    version="1.0.0",
    description="RAG interface for technical manuals backed by local embeddings, Chroma, and Ollama.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class RetrievedChunkPayload(BaseModel):
    id: str
    text: str
    distance: float
    metadata: dict


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved: list[RetrievedChunkPayload]


@app.get("/", response_class=FileResponse)
def read_index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(question, top_k=payload.top_k)
    except Exception as exc:
        logger.exception("Failed to answer question from web UI.")
        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate an answer right now: {exc}",
        ) from exc

    retrieved = [
        RetrievedChunkPayload(
            id=chunk.id,
            text=chunk.text,
            distance=chunk.distance,
            metadata=chunk.metadata or {},
        )
        for chunk in result.retrieved
    ]

    return AskResponse(
        question=result.question,
        answer=result.answer,
        retrieved=retrieved,
    )


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/ask/stream")
def ask_question_stream(payload: AskRequest) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def event_stream():
        try:
            retrieval = retrieve_context_for_question(question, top_k=payload.top_k)
            retrieved = [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "distance": chunk.distance,
                    "metadata": chunk.metadata or {},
                }
                for chunk in retrieval.retrieved
            ]
            yield sse_event(
                "retrieved",
                {
                    "question": retrieval.question,
                    "retrieved": retrieved,
                },
            )

            if retrieval.response_override is not None:
                yield sse_event(
                    "done",
                    {
                        "answer": retrieval.response_override,
                    },
                )
                return

            if not retrieval.context:
                yield sse_event(
                    "done",
                    {
                        "answer": "The answer is not explicitly found in the retrieved context.",
                    },
                )
                return

            client = OllamaLLMClient()
            full_answer = ""
            for piece in client.stream_answer(retrieval.question, retrieval.context):
                full_answer += piece
                yield sse_event("token", {"content": piece})

            yield sse_event("done", {"answer": full_answer.strip()})
        except Exception as exc:
            logger.exception("Failed to stream answer from web UI.")
            yield sse_event(
                "error",
                {"detail": f"Unable to generate an answer right now: {exc}"},
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
