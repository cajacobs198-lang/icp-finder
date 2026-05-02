import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .embed import get_embedder, firmographic_text


class Index:
    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame):
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        self.vectors = vectors.astype("float32")
        self.metadata = metadata.reset_index(drop=True)

    def save(self, dir_path: str | Path):
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        self.metadata.to_parquet(d / "metadata.parquet")
        (d / "meta.json").write_text(json.dumps({"n": len(self.vectors), "dim": int(self.vectors.shape[1])}))

    @classmethod
    def load(cls, dir_path: str | Path) -> "Index":
        d = Path(dir_path)
        v = np.load(d / "vectors.npy")
        m = pd.read_parquet(d / "metadata.parquet")
        return cls(v, m)

    @classmethod
    def build(cls, df: pd.DataFrame) -> "Index":
        """Embed every row of df and return a new Index."""
        embedder = get_embedder()
        texts = [firmographic_text(r) for r in df.to_dict(orient="records")]
        v = embedder.encode(texts)
        return cls(v, df)

    def search(self, query_vec: np.ndarray, top_k: int = 25, exclude: set[str] | None = None):
        """Return top_k (idx, score) sorted descending. Excludes by 'domain'."""
        # vectors are already normalized -> dot product == cosine
        sims = self.vectors @ query_vec
        order = np.argsort(-sims)
        out = []
        for i in order:
            domain = self.metadata.iloc[i].get("domain")
            if exclude and domain in exclude:
                continue
            out.append((int(i), float(sims[i])))
            if len(out) >= top_k:
                break
        return out
