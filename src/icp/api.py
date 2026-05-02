from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .index import Index
from .search import find

app = FastAPI(title="ICP Finder", version="0.1.0")
_index: Index | None = None


class FindReq(BaseModel):
    seeds: list[str]
    top: int = 25
    rerank: bool = False


@app.on_event("startup")
def _load():
    global _index
    try:
        _index = Index.load("index")
    except Exception:
        _index = None


@app.get("/health")
def health():
    return {"ok": _index is not None}


@app.post("/find")
def find_endpoint(req: FindReq):
    if _index is None:
        raise HTTPException(503, "index not built; run `icp build-index` first")
    return find(_index, [s.lower() for s in req.seeds], top=req.top, llm_rerank=req.rerank)
