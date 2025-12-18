from typing import List, Literal, Optional
import numpy as np

Metric = Literal["cosine", "euclidean"]


def _dist_to_point(U: np.ndarray, p: np.ndarray, metric: Metric) -> np.ndarray:
    # U: (N, D), p: (D,)
    if metric == "cosine":
        # U, p가 정규화되었다고 가정. 거리 = 1 - dot
        return 1.0 - U @ p
    # euclidean
    diff = U - p[None, :]
    return np.sqrt((diff * diff).sum(axis=1), dtype=np.float32)


def kcenter_greedy(
    U: np.ndarray,
    S: np.ndarray,
    k: int,
    metric: Metric = "cosine",
    precomputed_U_to_S: Optional[np.ndarray] = None,
) -> List[int]:
    """
    초기 커버 세트 S가 있을 때, U에서 k개를 추가로 선택해 커버리지를 극대화한다.
    반환: U 내에서 선택된 인덱스 리스트
    """
    n = U.shape[0]
    if n == 0 or k <= 0:
        return []
    if precomputed_U_to_S is not None and len(precomputed_U_to_S) == n:
        min_d = precomputed_U_to_S.copy()
    else:
        # S가 비었으면 무한대에서 시작
        if S.size == 0:
            min_d = np.full((n,), np.inf, dtype=np.float32)
        else:
            # U->S 최근접거리
            if metric == "cosine":
                min_d = (1.0 - U @ S.T).min(axis=1)
            else:
                # ||u-s|| 유클리드
                u2 = (U ** 2).sum(axis=1, keepdims=True)
                s2 = (S ** 2).sum(axis=1, keepdims=True).T
                dot = U @ S.T
                dist2 = np.maximum(u2 + s2 - 2.0 * dot, 0.0)
                min_d = np.sqrt(dist2, dtype=np.float32).min(axis=1)

    selected: List[int] = []
    for _ in range(min(k, n)):
        idx = int(np.argmax(min_d))
        selected.append(idx)
        # 새로 선택된 포인트와의 거리로 min_d 업데이트
        d_new = _dist_to_point(U, U[idx], metric=metric)
        min_d = np.minimum(min_d, d_new)
    return selected


