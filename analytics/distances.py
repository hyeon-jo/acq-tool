from typing import Literal, Tuple
import numpy as np

Metric = Literal["cosine", "euclidean"]


def _cosine_min_distances(U: np.ndarray, S: np.ndarray, chunk: int = 8192) -> np.ndarray:
    # U, S는 L2 정규화되어 있다고 가정 -> 거리 = 1 - dot
    m = U.shape[0]
    d_min = np.full((m,), np.inf, dtype=np.float32)
    for i in range(0, m, chunk):
        u = U[i:i + chunk]  # (c, d)
        # dot with S^T -> (c, ns)
        sim = u @ S.T
        dist = 1.0 - sim
        d_min[i:i + u.shape[0]] = np.minimum(d_min[i:i + u.shape[0]], dist.min(axis=1))
    return d_min


def _euclidean_min_distances(U: np.ndarray, S: np.ndarray, chunk: int = 4096) -> np.ndarray:
    m = U.shape[0]
    d_min = np.full((m,), np.inf, dtype=np.float32)
    for i in range(0, m, chunk):
        u = U[i:i + chunk]  # (c, d)
        # ||u - s|| = sqrt(||u||^2 + ||s||^2 - 2 u.s)
        u2 = (u ** 2).sum(axis=1, keepdims=True)  # (c,1)
        s2 = (S ** 2).sum(axis=1, keepdims=True).T  # (1,ns)
        dot = u @ S.T  # (c, ns)
        dist2 = np.maximum(u2 + s2 - 2.0 * dot, 0.0)
        dist = np.sqrt(dist2, dtype=np.float32)
        d_min[i:i + u.shape[0]] = np.minimum(d_min[i:i + u.shape[0]], dist.min(axis=1))
    return d_min


def min_distances_U_to_S(U: np.ndarray, S: np.ndarray, metric: Metric = "cosine") -> np.ndarray:
    if U.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if S.size == 0:
        return np.full((U.shape[0],), np.inf, dtype=np.float32)
    if metric == "cosine":
        return _cosine_min_distances(U, S)
    return _euclidean_min_distances(U, S)


def coverage_delta(dist_U_to_S: np.ndarray) -> float:
    if dist_U_to_S.size == 0:
        return 0.0
    return float(dist_U_to_S.max())


