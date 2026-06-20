import numpy as np
import hdbscan

# Embeddings are L2-normalised, so euclidean distances fall in [0, 2].
# `cluster_selection_epsilon` merges micro-clusters whose separation is below
# this distance — raising it makes the grouping less aggressive (fewer, larger
# groups). 0.0 keeps HDBSCAN's default, finer-grained splitting.
CLUSTER_SELECTION_EPSILON = 0.35


def cluster(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    cluster_selection_epsilon: float = CLUSTER_SELECTION_EPSILON,
) -> np.ndarray:

    # embeddings : (n, D)
    # return : (n,) of int labels or -1 -> singleton

    if len(embeddings) < 2:
        return np.full(len(embeddings), -1)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    return clusterer.fit_predict(embeddings)

