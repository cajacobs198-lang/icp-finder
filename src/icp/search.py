import os
import json
import numpy as np
from .index import Index
from .embed import get_embedder, firmographic_text


def centroid(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        raise ValueError("need at least one seed")
    c = vectors.mean(axis=0)
    n = np.linalg.norm(c)
    return c / (n if n else 1.0)


def seed_vectors(index: Index, seed_domains: list[str]) -> np.ndarray:
    rows = index.metadata
    mask = rows["domain"].isin([s.lower() for s in seed_domains])
    if not mask.any():
        raise ValueError("none of the seeds are in the index; build with a larger dataset")
    return index.vectors[mask.values]


def find(index: Index, seed_domains: list[str], top: int = 25, llm_rerank: bool = False) -> list[dict]:
    seeds = seed_vectors(index, seed_domains)
    q = centroid(seeds)
    candidates = index.search(q, top_k=top * (8 if llm_rerank else 1), exclude=set(seed_domains))
    rows = []
    for idx, score in candidates:
        r = index.metadata.iloc[idx].to_dict()
        rows.append({
            "domain": r.get("domain"),
            "name": r.get("name"),
            "industry": r.get("industry"),
            "country": r.get("country"),
            "score": round(score, 4),
            "why": f"{r.get('industry','?')}, {r.get('country','?')}, {r.get('employee_count','?')} employees",
        })
    if llm_rerank and os.environ.get("ANTHROPIC_API_KEY"):
        rows = _llm_rerank(rows, seed_domains)
    return rows[:top]


def _llm_rerank(rows: list[dict], seed_domains: list[str]) -> list[dict]:
    from anthropic import Anthropic  # noqa: WPS433
    client = Anthropic()
    prompt = (
        "You will re-rank candidate companies for similarity to a seed set of "
        f"happy customers: {', '.join(seed_domains)}. Return JSON list of "
        "{domain, score (0-1), why} sorted descending. Be ruthless: penalize "
        "obvious mismatches even if firmographics align.\n\nCANDIDATES:\n"
        + json.dumps(rows[:50])
    )
    msg = client.messages.create(
        model=os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    s, e = text.find("["), text.rfind("]")
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return rows
