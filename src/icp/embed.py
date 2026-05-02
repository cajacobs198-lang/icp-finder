import os
import numpy as np
from typing import Iterable


class LocalEmbedder:
    """sentence-transformers MiniLM. Lazy-loaded so the package imports fast."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        self._load()
        v = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(v, dtype="float32")


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        from openai import OpenAI  # noqa: WPS433
        client = OpenAI(api_key=self.api_key)
        rsp = client.embeddings.create(model=self.model, input=list(texts))
        v = np.asarray([d.embedding for d in rsp.data], dtype="float32")
        # L2-normalize for cosine via dot product
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms


def get_embedder():
    if os.environ.get("EMBED_PROVIDER", "local") == "openai":
        return OpenAIEmbedder(os.environ.get("EMBED_MODEL", "text-embedding-3-small"))
    return LocalEmbedder(os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2"))


def firmographic_text(row: dict) -> str:
    """Stable text representation of a company row used for embedding."""
    parts = [
        row.get("name", ""),
        f"industry: {row.get('industry', '')}",
        f"country: {row.get('country', '')}",
        f"employees: {row.get('employee_count', '')}",
        f"description: {row.get('description', '')}",
        "technologies: " + ", ".join(row.get("technologies", []) or []),
    ]
    return " | ".join(p for p in parts if p)
