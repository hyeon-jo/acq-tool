from typing import Optional
import numpy as np
from sklearn.decomposition import PCA
import umap


def reduce_umap_3d(
    embeddings: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    use_pca_first: bool = False,
    pca_dim: int = 64,
    random_state: int = 42,
) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    X = embeddings
    if use_pca_first and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    out = reducer.fit_transform(X)
    return out.astype(np.float32)


