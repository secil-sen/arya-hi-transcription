import numpy as np
from typing import List

####################### Embedding Utils #######################

def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n

def duration_weighted_mean_np(vecs: List[np.ndarray], durs: List[float]) -> np.ndarray:
    w = np.asarray(durs, dtype=np.float32)
    w = w / (w.sum() + 1e-12)
    V = np.vstack(vecs).astype(np.float32)  # [N, D]
    return (w[:, None] * V).sum(axis=0)

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return -1.0
    return float(np.dot(a, b) / (na * nb))