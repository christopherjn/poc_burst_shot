import numpy as np
import hdbscan

def cluster(embeddings: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:

    # embeddings : (n, D)
    # return : (n,) of int labels or -1 -> singleton

    if len(embeddings) < 2:
        return np.full(len(embeddings), -1)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    return clusterer.fit_predict(embeddings)

