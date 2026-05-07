import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def _flatten_features(x):
    x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError("Expected at least 2D array, got shape %s" % (x.shape,))
    return x.reshape(x.shape[0], -1)


def _variable_radius_neighbors(tree, queries, radii):
    max_radius = float(np.max(radii)) if len(radii) > 0 else 0.0
    distances_all, indices_all = tree.radius_neighbors(
        queries,
        radius=max_radius,
        return_distance=True,
        sort_results=True,
    )
    filtered = []
    for i in range(len(queries)):
        keep = distances_all[i] < radii[i]
        filtered.append(indices_all[i][keep])
    return filtered


def _hopcroft_karp(adjacency, num_left, num_right):
    matched_left = np.full(num_left, -1, dtype=int)
    matched_right = np.full(num_right, -1, dtype=int)
    dist = np.zeros(num_left, dtype=int)

    def bfs():
        queue = []
        for u in range(num_left):
            if matched_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1

        found_augmenting_path = False
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adjacency[u]:
                mate = matched_right[v]
                if mate == -1:
                    found_augmenting_path = True
                elif dist[mate] == -1:
                    dist[mate] = dist[u] + 1
                    queue.append(mate)
        return found_augmenting_path

    def dfs(u):
        for v in adjacency[u]:
            mate = matched_right[v]
            if mate == -1 or (dist[mate] == dist[u] + 1 and dfs(mate)):
                matched_left[u] = v
                matched_right[v] = u
                return True
        dist[u] = -1
        return False

    matching_size = 0
    while bfs():
        for u in range(num_left):
            if matched_left[u] == -1 and dfs(u):
                matching_size += 1

    return matching_size, matched_left, matched_right


def compute_capacity_recall(
    real_features,
    fake_features,
    nearest_k=5,
    n_jobs=8,
    radius_mode="min_mean",
    matching_backend="scipy",
):
    """
    Capacity-constrained recall via maximum bipartite matching.

    A real sample i defines an admissible radius.

    radius_mode:
        - "local": epsilon_i = real-kNN-radius_i
        - "min_mean": epsilon_i = min(real-kNN-radius_i, mean(real-kNN-radius))
        - "min_median": epsilon_i = min(real-kNN-radius_i, median(real-kNN-radius))

    A fake sample j can explain real sample i iff:
        ||g_j - r_i|| < epsilon_i
    """
    real_flat = _flatten_features(real_features)
    fake_flat = _flatten_features(fake_features)

    num_real = real_flat.shape[0]
    num_fake = fake_flat.shape[0]

    if num_real == 0 or num_fake == 0:
        return {
            "summary": {
                "matching_recall": 0.0,
                "reachable_real_ratio": 0.0,
                "under_representation_ratio": 0.0,
                "over_representation_ratio": 0.0,
            }
        }

    real_tree = NearestNeighbors(
        n_neighbors=min(nearest_k + 1, num_real),
        algorithm="ball_tree",
        n_jobs=n_jobs,
    ).fit(real_flat)
    fake_tree = NearestNeighbors(
        algorithm="ball_tree",
        n_jobs=n_jobs,
    ).fit(fake_flat)

    distances_real, _ = real_tree.kneighbors(real_flat)
    real_knn_radius = distances_real[:, -1]
    global_mean_radius = float(np.mean(real_knn_radius))
    global_median_radius = float(np.median(real_knn_radius))

    if radius_mode == "local":
        effective_radius = real_knn_radius.copy()
    elif radius_mode == "min_mean":
        effective_radius = np.minimum(real_knn_radius, global_mean_radius)
    elif radius_mode == "min_median":
        effective_radius = np.minimum(real_knn_radius, global_median_radius)
    else:
        raise ValueError(
            f"Unknown radius_mode={radius_mode!r}. Expected 'local', 'min_mean', or 'min_median'."
        )

    fake_neighbors_per_real = _variable_radius_neighbors(
        fake_tree,
        real_flat,
        effective_radius,
    )

    adjacency = []
    for real_idx in range(num_real):
        neighbor_ids = fake_neighbors_per_real[real_idx]
        if len(neighbor_ids) == 0:
            adjacency.append(np.array([], dtype=int))
            continue
        adjacency.append(np.asarray(neighbor_ids, dtype=int))

    if matching_backend == "hk":
        matched_count, matched_left, matched_right = _hopcroft_karp(
            adjacency,
            num_left=num_real,
            num_right=num_fake,
        )
    elif matching_backend == "scipy":
        row_idx = []
        col_idx = []
        for r, neigh in enumerate(adjacency):
            if len(neigh) == 0:
                continue
            row_idx.extend([r] * len(neigh))
            col_idx.extend(np.asarray(neigh, dtype=int).tolist())

        if len(row_idx) == 0:
            matched_left = np.full(num_real, -1, dtype=int)
            matched_right = np.full(num_fake, -1, dtype=int)
            matched_count = 0
        else:
            data = np.ones(len(row_idx), dtype=np.int8)
            biadj = csr_matrix((data, (row_idx, col_idx)), shape=(num_real, num_fake))
            # right_to_left[fake_idx] = real_idx or -1
            right_to_left = maximum_bipartite_matching(biadj, perm_type="column")
            matched_right = np.asarray(right_to_left, dtype=int)
            matched_left = np.full(num_real, -1, dtype=int)
            for fake_idx, real_idx in enumerate(matched_right):
                if real_idx != -1:
                    matched_left[real_idx] = fake_idx
            matched_count = int(np.sum(matched_right != -1))
    else:
        raise ValueError(
            f"Unknown matching_backend={matching_backend!r}. Expected 'hk' or 'scipy'."
        )

    reachable_real_mask = np.array(
        [len(neighbor_ids) > 0 for neighbor_ids in fake_neighbors_per_real],
        dtype=bool,
    )
    reachable_real_count = int(np.sum(reachable_real_mask))

    reachable_fake_mask = np.zeros(num_fake, dtype=bool)
    for neighbor_ids in fake_neighbors_per_real:
        if len(neighbor_ids) > 0:
            reachable_fake_mask[np.asarray(neighbor_ids, dtype=int)] = True
    reachable_fake_count = int(np.sum(reachable_fake_mask))

    matched_real_count = int(matched_count)
    matched_fake_count = int(matched_count)

    matching_recall = float(matched_real_count / num_real)
    reachable_real_ratio = float(reachable_real_count / num_real)
    under_representation_ratio = float(
        (reachable_real_count - matched_real_count) / num_real
    )
    over_representation_ratio = float(
        (reachable_fake_count - matched_fake_count) / num_fake
    )

    summary = {
        "matching_recall": matching_recall,
        "reachable_real_ratio": reachable_real_ratio,
        "under_representation_ratio": under_representation_ratio,
        "over_representation_ratio": over_representation_ratio,
    }

    return {"summary": summary}



def compute_capacity_aware_distribution_matching_analysis(
    real_features,
    fake_features,
    nearest_k=5,
    n_jobs=8,
    radius_mode="min_mean",
    matching_backend="scipy",
):
    """
    Unified method wrapper for Capacity-Aware Distribution Matching Analysis.

    Returns only the CADMA scores used by the Gaussian artifact.
    """
    res = compute_capacity_recall(
        real_features=real_features,
        fake_features=fake_features,
        nearest_k=nearest_k,
        n_jobs=n_jobs,
        radius_mode=radius_mode,
        matching_backend=matching_backend,
    )
    summary = res["summary"]
    metrics = {
        "CapacityRecall": summary["matching_recall"],
        "ReachableRealRatio": summary["reachable_real_ratio"],
        "UnderRepresentationRatio": summary["under_representation_ratio"],
        "OverRepresentationRatio": summary["over_representation_ratio"],
    }
    return {"metrics": metrics}
