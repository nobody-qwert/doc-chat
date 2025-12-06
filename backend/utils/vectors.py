from __future__ import annotations

import numpy as np


def cosine_similarity(a, b) -> float:
    vector_a = a.astype("float32").ravel()
    vector_b = b.astype("float32").ravel()
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denom)
