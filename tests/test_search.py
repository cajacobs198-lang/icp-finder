import numpy as np
import pandas as pd
from icp.index import Index
from icp.search import centroid, find


def _toy_index() -> Index:
    # 3 clusters of 2 vectors each, 4-dim, normalized
    raw = np.array([
        [1, 0, 0, 0],  # A1
        [0.99, 0.05, 0.0, 0.0],  # A2 (near A1)
        [0, 1, 0, 0],  # B1
        [0.05, 0.99, 0, 0],  # B2
        [0, 0, 1, 0],  # C1
        [0, 0, 0.99, 0.05],  # C2
    ], dtype="float32")
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    meta = pd.DataFrame({
        "domain": [f"d{i}.com" for i in range(6)],
        "name": ["A1", "A2", "B1", "B2", "C1", "C2"],
        "industry": ["A", "A", "B", "B", "C", "C"],
        "country": ["US"] * 6,
        "employee_count": [100] * 6,
    })
    return Index(raw, meta)


def test_centroid_normalized():
    v = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
    c = centroid(v)
    assert abs(np.linalg.norm(c) - 1.0) < 1e-6


def test_find_returns_cluster_neighbor_excluding_seeds():
    idx = _toy_index()
    out = find(idx, ["d0.com"], top=2)
    domains = [r["domain"] for r in out]
    assert "d0.com" not in domains
    assert "d1.com" == domains[0]  # nearest neighbor in same cluster


def test_find_with_multi_seed_centroid():
    idx = _toy_index()
    out = find(idx, ["d0.com", "d2.com"], top=2)
    # centroid is between A and B, so top should be from A or B, not C
    industries = {r["industry"] for r in out}
    assert industries.issubset({"A", "B"})
