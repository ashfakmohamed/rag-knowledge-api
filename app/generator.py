import os
from typing import Protocol

import httpx

from app.retriever import StoredDocument


class AnswerGenerator(Protocol):
    async def generate(
        self, question: str, contexts: list[tuple[StoredDocument, float]]
    ) -> str: ...


class ExtractiveGenerator:
    """Secret-free fallback that produces an answer only from retrieved context."""

    async def generate(
        self, question: str, contexts: list[tuple[StoredDocument, float]]
    ) -> str:
        del question
        if not contexts:
            return "I could not find relevant context for that question."
        excerpts = [document.content.strip()[:320] for document, _ in contexts]
        return " ".join(excerpts)


class OpenAICompatibleGenerator:
    """Optional adapter for any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model

    async def generate(
        self, question: str, contexts: list[tuple[StoredDocument, float]]
    ) -> str:
        context = "\n\n".join(
            f"[{index}] {document.title}\n{document.content}"
            for index, (document, _) in enumerate(contexts, start=1)
        )
        prompt = (
            "Answer only from the supplied context. If the context is insufficient, say so. "
            "Cite supporting passages with bracketed numbers.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a grounded knowledge assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


def generator_from_environment() -> AnswerGenerator:
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if base_url and api_key and model:
        return OpenAICompatibleGenerator(base_url, api_key, model)
    return ExtractiveGenerator()
