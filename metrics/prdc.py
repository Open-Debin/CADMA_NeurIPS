import numpy as np
from sklearn.neighbors import NearestNeighbors


def _effective_n_jobs(n_jobs, n_samples):
    if n_samples < 2000:
        return 1
    return max(1, int(n_jobs))


def _safe_bincount(index_lists, minlength):
    non_empty = [idx for idx in index_lists if len(idx) > 0]
    if not non_empty:
        return np.zeros(minlength, dtype=int)
    return np.bincount(np.concatenate(non_empty), minlength=minlength)


class NeighborCache:
    def __init__(self, real_features, fake_features, n_jobs=8):
        self.real_features = real_features
        self.fake_features = fake_features
        self.N = real_features.shape[0]
        self.M = fake_features.shape[0]
        self.real_flat = real_features.reshape(self.N, -1)
        self.fake_flat = fake_features.reshape(self.M, -1)
        self.n_jobs = _effective_n_jobs(n_jobs, min(self.N, self.M))
        self.low_parallel = self.n_jobs <= 4
        self._radii_cache = {}
        self._membership_cache = {}

        # Fit trees once and reuse for all metrics.
        self.real_tree = NearestNeighbors(
            algorithm="ball_tree",
            n_jobs=(1 if self.low_parallel else self.n_jobs),
        ).fit(self.real_flat)
        self.fake_tree = NearestNeighbors(
            algorithm="ball_tree",
            n_jobs=(1 if self.low_parallel else self.n_jobs),
        ).fit(self.fake_flat)

    def self_radii(self, domain, nearest_k):
        key = (domain, nearest_k)
        if key in self._radii_cache:
            return self._radii_cache[key]

        if domain == "real":
            distances, _ = self.real_tree.kneighbors(
                self.real_flat, n_neighbors=nearest_k + 1
            )
        elif domain == "fake":
            distances, _ = self.fake_tree.kneighbors(
                self.fake_flat, n_neighbors=nearest_k + 1
            )
        else:
            raise ValueError("Unknown domain: %s" % domain)
        radii = distances[:, -1]
        self._radii_cache[key] = radii
        return radii

    def _radius_neighbors_variable(self, tree, queries, radii):
        if self.low_parallel:
            return tree.radius_neighbors(
                queries, radius=radii, return_distance=False
            )

        distances, candidates = tree.radius_neighbors(
            queries, radius=np.max(radii), return_distance=True
        )
        filtered = np.empty(len(queries), dtype=np.ndarray)
        for i in range(len(queries)):
            filtered[i] = candidates[i][distances[i] <= radii[i]]
        return filtered

    def memberships(self, query_domain, center_domain, radius_domain, nearest_k):
        key = (query_domain, center_domain, radius_domain, nearest_k)
        if key in self._membership_cache:
            return self._membership_cache[key]

        if query_domain == "real":
            queries = self.real_flat
        elif query_domain == "fake":
            queries = self.fake_flat
        else:
            raise ValueError("Unknown query domain: %s" % query_domain)

        if center_domain == "real":
            tree = self.real_tree
        elif center_domain == "fake":
            tree = self.fake_tree
        else:
            raise ValueError("Unknown center domain: %s" % center_domain)

        radii = self.self_radii(radius_domain, nearest_k)
        members = self._radius_neighbors_variable(tree, queries, radii)
        self._membership_cache[key] = members
        return members


def _radius_neighbors_variable(tree, queries, radii, low_parallel):
    if low_parallel:
        return tree.radius_neighbors(
            queries,
            radius=radii,
            return_distance=False,
        )

    distances_candidates, indices_candidates = tree.radius_neighbors(
        queries,
        radius=np.max(radii),
        return_distance=True,
    )
    filtered_indices = np.empty(len(queries), dtype=np.ndarray)
    for i in range(len(queries)):
        valid = distances_candidates[i] <= radii[i]
        filtered_indices[i] = indices_candidates[i][valid]
    return filtered_indices


def _local_balance_score(real_count, fake_count, exponent):
    if not (0.0 < exponent < 1.0):
        raise ValueError("match_exponent must be in (0, 1), got %s" % exponent)

    total = real_count + fake_count
    x = real_count / total
    denom = (2.0 ** (1.0 - exponent)) - 1.0
    score = ((x**exponent) + ((1.0 - x) ** exponent) - 1.0) / denom
    return np.clip(score, 0.0, 1.0)


def prdc(
    real_features,
    fake_features,
    nearest_k,
    sym=False,
    ppr=False,
    match=False,
    match_fixed_radius=False,
    match_minavg_radius=False,
    ratio_match=False,
    a_ppr=1.2,
    match_exponent=0.5,
    n_jobs=8,
    cache=None,
):
    """
    Computes precision, recall, density, coverage, and optionally symmetric
    precision & recall, P-precision & P-recall, and local distribution-match
    metrics.

    Args:
        real_features (np.ndarray): real features
        fake_features (np.ndarray): fake features
        nearest_k (int): number of nearest neighbors to consider
        sym (bool, optional): whether to compute symmetric precision and
            recall. Defaults to False.
        ppr (bool, optional): whether to compute P-precision and P-recall.
            Defaults to False.
        match (bool, optional): whether to compute the proposed local
            real/fake proportion match metrics. Defaults to False.
        match_exponent (float, optional): exponent `a` used in the local
            balance scoring rule. Defaults to 0.5.
        n_jobs (int, optional): number of jobs to run in parallel. Defaults
            to 8.

    Returns:
        dictionary containing precision, recall, density, coverage and
        optionally symmetric precision and recall.
    """
    if cache is None:
        cache = NeighborCache(real_features, fake_features, n_jobs=n_jobs)

    N = cache.N
    M = cache.M
    real_features_flat = cache.real_flat
    fake_features_flat = cache.fake_flat
    radii_real = cache.self_radii("real", nearest_k)
    radii_fake = cache.self_radii("fake", nearest_k)
    low_parallel = cache.low_parallel
    real_tree = cache.real_tree
    fake_tree = cache.fake_tree

    all_fake_in_real_balls = cache.memberships(
        query_domain="real",
        center_domain="fake",
        radius_domain="real",
        nearest_k=nearest_k,
    )

    # Precision: is each fake point inside a real ball?
    # radii_real are the radii of the real balls
    # all_fake_in_real_balls is the list of fake points inside each real ball
    fake_appearance_count = _safe_bincount(all_fake_in_real_balls, minlength=M)
    # fake_appearance_count[i] is the number of real balls the fake point
    # i is in
    precision = np.mean(fake_appearance_count > 0)

    # Density is the average number of real balls each fake point is in
    density = np.mean(fake_appearance_count) / nearest_k

    # Coverage: is there a fake point inside each real ball?
    coverage = np.mean(
        [
            len(fake_in_real_ball_i) > 0
            for fake_in_real_ball_i in all_fake_in_real_balls
        ]
    )

    all_real_in_fake_balls = cache.memberships(
        query_domain="fake",
        center_domain="real",
        radius_domain="fake",
        nearest_k=nearest_k,
    )
    # Recall: is each real point inside a fake ball?
    # radii_fake are the radii of the fake balls
    # all_real_in_fake_balls is the list of real points inside each fake ball
    real_appearance_count = _safe_bincount(all_real_in_fake_balls, minlength=N)
    # real_appearance_count[i] is the number of fake balls the real point i is
    # in
    recall = np.mean(real_appearance_count > 0)

    res = dict(
        Precision=precision,
        Recall=recall,
        Density=density,
        Coverage=coverage,
    )

    if sym:
        symRecall = np.min((recall, coverage))

        # cPrecision is the symmetric of coverage: is there a real point in
        # each fake ball?
        cPrecision = np.mean(
            [
                len(real_in_fake_ball_i) > 0
                for real_in_fake_ball_i in all_real_in_fake_balls
            ]
        )

        symPrecision = np.min((precision, cPrecision))

        res["symRecall"] = symRecall
        res["symPrecision"] = symPrecision

    if ppr:
        R_precision = np.mean(radii_real) * a_ppr
        R_recall = np.mean(radii_fake) * a_ppr

        if low_parallel and n_jobs > 1:
            # Can be done in parallel because the radius is the same for all
            # points, so build the trees again with n_jobs>1
            fake_tree = NearestNeighbors(
                n_neighbors=nearest_k,
                algorithm="ball_tree",
                n_jobs=cache.n_jobs,
            ).fit(fake_features_flat)
            real_tree = NearestNeighbors(
                n_neighbors=nearest_k,
                algorithm="ball_tree",
                n_jobs=cache.n_jobs,
            ).fit(real_features_flat)

        distances_real_in_fake_balls, _all_real_in_fake_balls_ppr = (
            real_tree.radius_neighbors(
                fake_features_flat,
                radius=R_precision,
                return_distance=True,
            )
        )
        prods = np.ones(M)
        for j in range(M):
            if len(distances_real_in_fake_balls[j]) > 0:
                prods[j] = np.prod(
                    distances_real_in_fake_balls[j] / R_precision
                )

        # P-precision is the average over all fake points of the product of
        # the probability over real points that y_j does not belong to the
        # support around x_i
        res["P-precision"] = np.mean(1 - prods)

        distances_fake_in_real_balls, _all_fake_in_real_balls_ppr = (
            fake_tree.radius_neighbors(
                real_features_flat,
                radius=R_recall,
                return_distance=True,
            )
        )
        prods = np.ones(N)
        for i in range(N):
            if len(distances_fake_in_real_balls[i]) > 0:
                prods[i] = np.prod(distances_fake_in_real_balls[i] / R_recall)
        res["P-recall"] = np.mean(1 - prods)

    if match:
        all_real_in_real_balls = _radius_neighbors_variable(
            cache.real_tree,
            real_features_flat,
            radii_real,
            low_parallel=low_parallel,
        )
        all_fake_in_fake_balls = _radius_neighbors_variable(
            cache.fake_tree,
            fake_features_flat,
            radii_fake,
            low_parallel=low_parallel,
        )

        real_anchor_scores = np.array(
            [
                _local_balance_score(
                    real_count=len(all_real_in_real_balls[i]),
                    fake_count=len(all_fake_in_real_balls[i]),
                    exponent=match_exponent,
                )
                for i in range(N)
            ]
        )
        fake_anchor_scores = np.array(
            [
                _local_balance_score(
                    real_count=len(all_real_in_fake_balls[j]),
                    fake_count=len(all_fake_in_fake_balls[j]),
                    exponent=match_exponent,
                )
                for j in range(M)
            ]
        )

        # Real-anchored score measures how well fake samples match local real
        # regions, so it behaves like a recall-style metric.
        res["MatchRecall"] = np.mean(real_anchor_scores)
        # Fake-anchored score measures how well fake regions are proportionally
        # matched by real samples, so it behaves like a precision-style metric.
        res["MatchPrecision"] = np.mean(fake_anchor_scores)

    if match_fixed_radius:
        # Fixed radius estimated once from real data only:
        # r = mean distance to kth nearest neighbor among real samples.
        fixed_radius = float(np.mean(radii_real))

        all_real_in_real_fixed = cache.real_tree.radius_neighbors(
            real_features_flat,
            radius=fixed_radius,
            return_distance=False,
        )
        all_fake_in_real_fixed = cache.fake_tree.radius_neighbors(
            real_features_flat,
            radius=fixed_radius,
            return_distance=False,
        )
        all_real_in_fake_fixed = cache.real_tree.radius_neighbors(
            fake_features_flat,
            radius=fixed_radius,
            return_distance=False,
        )
        all_fake_in_fake_fixed = cache.fake_tree.radius_neighbors(
            fake_features_flat,
            radius=fixed_radius,
            return_distance=False,
        )

        real_anchor_scores_fixed = np.array(
            [
                _local_balance_score(
                    real_count=len(all_real_in_real_fixed[i]),
                    fake_count=len(all_fake_in_real_fixed[i]),
                    exponent=match_exponent,
                )
                for i in range(N)
            ]
        )
        fake_anchor_scores_fixed = np.array(
            [
                _local_balance_score(
                    real_count=len(all_real_in_fake_fixed[j]),
                    fake_count=len(all_fake_in_fake_fixed[j]),
                    exponent=match_exponent,
                )
                for j in range(M)
            ]
        )

        res["MatchRecallFixedR"] = np.mean(real_anchor_scores_fixed)
        res["MatchPrecisionFixedR"] = np.mean(fake_anchor_scores_fixed)

    if match_minavg_radius:
        # Radius per point = min(local kNN radius, global average kNN radius).
        # Keeps dense regions local while capping overly large sparse regions.
        mean_r_real = float(np.mean(radii_real))
        mean_r_fake = float(np.mean(radii_fake))
        radii_real_minavg = np.minimum(radii_real, mean_r_real)
        radii_fake_minavg = np.minimum(radii_fake, mean_r_fake)

        all_real_in_real_minavg = _radius_neighbors_variable(
            cache.real_tree,
            real_features_flat,
            radii_real_minavg,
            low_parallel=low_parallel,
        )
        all_fake_in_real_minavg = _radius_neighbors_variable(
            cache.fake_tree,
            real_features_flat,
            radii_real_minavg,
            low_parallel=low_parallel,
        )
        all_real_in_fake_minavg = _radius_neighbors_variable(
            cache.real_tree,
            fake_features_flat,
            radii_fake_minavg,
            low_parallel=low_parallel,
        )
        all_fake_in_fake_minavg = _radius_neighbors_variable(
            cache.fake_tree,
            fake_features_flat,
            radii_fake_minavg,
            low_parallel=low_parallel,
        )

        real_anchor_scores_minavg = np.array(
            [
                _local_balance_score(
                    real_count=len(all_real_in_real_minavg[i]),
                    fake_count=len(all_fake_in_real_minavg[i]),
                    exponent=match_exponent,
                )
                for i in range(N)
            ]
        )
        fake_anchor_scores_minavg = np.array(
            [
                _local_balance_score(
                    real_count=len(all_real_in_fake_minavg[j]),
                    fake_count=len(all_fake_in_fake_minavg[j]),
                    exponent=match_exponent,
                )
                for j in range(M)
            ]
        )

        res["MatchRecallMinAvgR"] = np.mean(real_anchor_scores_minavg)
        res["MatchPrecisionMinAvgR"] = np.mean(fake_anchor_scores_minavg)

    if ratio_match:
        # Simpler ratio-based variant:
        # x := real / (real + fake) in each local neighborhood.
        # Precision uses fake-anchored neighborhoods and averages x.
        # Recall uses real-anchored neighborhoods and averages (1 - x).
        all_real_in_real_balls = _radius_neighbors_variable(
            cache.real_tree,
            real_features_flat,
            radii_real,
            low_parallel=low_parallel,
        )
        all_fake_in_fake_balls = _radius_neighbors_variable(
            cache.fake_tree,
            fake_features_flat,
            radii_fake,
            low_parallel=low_parallel,
        )

        # Precision-style anchor: iterate fake points, measure local real ratio.
        ratio_precision_vals = []
        for j in range(M):
            real_count = len(all_real_in_fake_balls[j])
            fake_count = len(all_fake_in_fake_balls[j])
            total = real_count + fake_count
            ratio_precision_vals.append(
                0.0 if total == 0 else real_count / total
            )

        # Recall-style anchor: iterate real points, measure local fake ratio.
        ratio_recall_vals = []
        for i in range(N):
            real_count = len(all_real_in_real_balls[i])
            fake_count = len(all_fake_in_real_balls[i])
            total = real_count + fake_count
            ratio_recall_vals.append(
                0.0 if total == 0 else fake_count / total
            )

        res["RatioPrecision"] = np.mean(ratio_precision_vals)
        res["RatioRecall"] = np.mean(ratio_recall_vals)

    return res


def _precision_recall_cover_direction(
    P_flat,
    Q_flat,
    P_tree,
    Q_tree,
    k=3,
    C=3,
    n_jobs=8,
):
    k_prime = C * k
    n_jobs = _effective_n_jobs(n_jobs, min(len(P_flat), len(Q_flat)))

    low_parallel = n_jobs <= 4

    # Compute k' nearest neighbors in Q for each point in Q
    distances_k_prime, _ = Q_tree.kneighbors(Q_flat, n_neighbors=k_prime)
    r_Q = distances_k_prime[:, -1]

    # For each point in Q, find the indices of the points in P that are within
    # the corresponding radius r_Q
    if low_parallel:
        indices = P_tree.radius_neighbors(
            Q_flat, radius=r_Q, return_distance=False
        )
    else:
        dists, indices_candidates = P_tree.radius_neighbors(
            Q_flat, radius=np.max(r_Q), return_distance=True
        )
        indices = [
            idx[dists[i] <= r_Q[i]] for i, idx in enumerate(indices_candidates)
        ]

    is_covered = np.array([len(idx) >= k for idx in indices])
    covered_count = np.sum(is_covered)

    PC = covered_count / len(Q_flat)
    return PC


def precision_recall_cover_pair(
    real_features,
    fake_features,
    k=3,
    C=3,
    n_jobs=8,
    cache=None,
):
    if cache is None:
        cache = NeighborCache(real_features, fake_features, n_jobs=n_jobs)

    precision_cover = _precision_recall_cover_direction(
        P_flat=cache.real_flat,
        Q_flat=cache.fake_flat,
        P_tree=cache.real_tree,
        Q_tree=cache.fake_tree,
        k=k,
        C=C,
        n_jobs=cache.n_jobs,
    )
    recall_cover = _precision_recall_cover_direction(
        P_flat=cache.fake_flat,
        Q_flat=cache.real_flat,
        P_tree=cache.fake_tree,
        Q_tree=cache.real_tree,
        k=k,
        C=C,
        n_jobs=cache.n_jobs,
    )
    return precision_cover, recall_cover


def precision_recall_cover_fast(P, Q, k=3, C=3, n_jobs=8):
    # Kept for backward compatibility; one-direction PrecisionCover only.
    cache = NeighborCache(P, Q, n_jobs=n_jobs)
    return _precision_recall_cover_direction(
        P_flat=cache.real_flat,
        Q_flat=cache.fake_flat,
        P_tree=cache.real_tree,
        Q_tree=cache.fake_tree,
        k=k,
        C=C,
        n_jobs=cache.n_jobs,
    )
