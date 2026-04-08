import json

import requests

from genai_tech_assistant.config import settings
from genai_tech_assistant.logging_config import get_logger

logger = get_logger(__name__)


class OllamaLLMClient:

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model_name or settings.ollama_model_name

        logger.info(
            f"Initialized OllamaLLMClient with model={self.model_name} "
            f"base_url={self.base_url}"
        )

    def generate_answer(self, question: str, context: str) -> str:
        payload = self._build_payload(question, context, stream=False)

        logger.info(
            f"Calling Ollama model={self.model_name} "
            f"(question length={len(question)}, context length={len(context)})"
        )

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"].strip()

    def stream_answer(self, question: str, context: str):
        payload = self._build_payload(question, context, stream=True)

        logger.info(
            f"Streaming Ollama model={self.model_name} "
            f"(question length={len(question)}, context length={len(context)})"
        )

        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
            stream=True,
        ) as response:
            response.raise_for_status()

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                data = json.loads(raw_line)
                message = data.get("message") or {}
                content = message.get("content", "")
                if content:
                    yield content

                if data.get("done"):
                    break

    def _build_payload(self, question: str, context: str, stream: bool) -> dict:
        system_prompt = """
You are a technician assistant for industrial equipment documentation.

Use ONLY the provided context.

Core rules:
1. Never guess.
2. Never infer a fact that is not explicitly supported by the context.
3. If the answer is not clearly present in the context, say so directly.
4. Be practical, concise, and technician-friendly.
5. Prefer exact evidence over broad interpretation.
6. If multiple retrieved chunks are relevant, combine them carefully.
7. Do NOT mention chunk numbers, retrieval ranks, similarity distances, or internal retrieval process.
8. Do NOT output raw metadata strings such as source=..., page=..., distance=....
9. If useful, you may end with a short source line like:
   Source: <file name>, page <page number>.
10. If a count and a visible list do not fully match, do not invent missing items. Report only what is explicitly visible.
11. If a count and a visible list do not fully match, do not invent missing items. Report only what is explicitly visible.

12. If the context states a total count but only some items are explicitly visible, do not present the incomplete list as complete. Say that only the visible items are listed in the retrieved context.
Response behavior by question type:

A) Troubleshooting / fault / diagnostic questions
- Start with the direct answer if present.
- Then provide:
  1. What it means
  2. What to check next
  3. Safety note (only if relevant from context)

B) Overview / structure / definition / component questions
- Start with the direct answer.
- Then provide a short explanation or list if the context supports it.

C) Missing or weak evidence
- Say clearly: "The answer is not explicitly found in the retrieved context."
- Then briefly say what kind of section or document should be checked next.
- Do not invent likely meanings from similar terms or nearby codes.

Formatting rules:
- Use short headings when helpful.
- Use bullet points or numbered lists only when they improve clarity.
- Do not mention that you are an AI assistant.
- Do not say "according to chunk..." or anything similar.
- Do not use Markdown formatting like **bold** or # headings.
"""

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer using only the retrieved context."
        )

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
        }
