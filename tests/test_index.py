import numpy as np
import pandas as pd
from icp.index import Index


def test_round_trip_save_load(tmp_path):
    v = np.eye(4, dtype="float32")
    m = pd.DataFrame({"domain": [f"d{i}.com" for i in range(4)], "name": list("abcd")})
    idx = Index(v, m)
    idx.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert (loaded.vectors == v).all()
    assert list(loaded.metadata["domain"]) == list(m["domain"])
