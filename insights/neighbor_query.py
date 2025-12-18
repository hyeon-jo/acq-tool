from typing import List, Tuple, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors


def query_neighbors(
    query_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_paths: List[str],
    top_k: int = 8,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    3D 좌표계에서 최근접 이웃 검색.
    반환: (거리배열 shape=(top_k,), 인덱스 목록, 경로 목록)
    """
    if candidate_xyz.size == 0 or len(candidate_paths) == 0:
        return np.array([]), [], []
    nns = min(top_k, len(candidate_paths))
    # 대규모일 때 FAISS 우선 사용(있다면)
    try:
        import faiss  # type: ignore
        xb = candidate_xyz.astype(np.float32)
        xq = query_xyz.astype(np.float32)
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(xb)
        D, I = index.search(xq, nns)
        idx = I[0].tolist()
        return D[0], idx, [candidate_paths[i] for i in idx]
    except Exception:
        nn = NearestNeighbors(n_neighbors=nns, metric="euclidean")
        nn.fit(candidate_xyz)
        dist, idx = nn.kneighbors(query_xyz, return_distance=True)
        idx = idx[0].tolist()
        return dist[0], idx, [candidate_paths[i] for i in idx]


