import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.special import gammaln, betaln, comb


class ClippedDensityCoverage:
    def __init__(
        self,
        real_data,
        K=5,
        n_jobs=8,
    ):
        self.real_data_neighbors = real_data.reshape(real_data.shape[0], -1)
        self.K = K
        self.n_jobs = n_jobs

        self.f_coverage_values = None

        self.distances_real = None
        self.indices_real = None

    def ClippedDensity(self, synthetic_data, can_exceed_one=False):
        synthetic_data_neighbors = synthetic_data.reshape(
            synthetic_data.shape[0], -1
        )

        if self.distances_real is None:
            neighbors_real = NearestNeighbors(
                n_neighbors=1 + self.K, n_jobs=self.n_jobs
            ).fit(self.real_data_neighbors)
            distances_real, indices_real = neighbors_real.kneighbors(
                self.real_data_neighbors
            )
            self.distances_real = distances_real
            self.indices_real = indices_real

        distances_real_K = self.distances_real[:, -1]  # radius_K
        indices_real = self.indices_real[:, 1:]  # remove the point itself

        # 1.5 fix the radius
        distances_real_K = np.clip(
            distances_real_K,
            0,
            np.median(distances_real_K),
        )
        # mask the indices of real points to keep only the ones still in the
        # ball
        mask = self.distances_real[:, 1:] <= distances_real_K[:, np.newaxis]
        ind_real = indices_real[mask]

        # 2. For each real point, find all the synthetic points in its ball
        tree_synthetic = NearestNeighbors(
            algorithm="ball_tree", metric="euclidean", n_jobs=self.n_jobs
        ).fit(synthetic_data_neighbors)

        distances_to_synthetic, indices_to_synthetic = (
            tree_synthetic.radius_neighbors(
                X=self.real_data_neighbors,
                radius=np.max(distances_real_K),
                return_distance=True,
                sort_results=False,
            )
        )

        ind_synthetic = [
            indices_to_synthetic[i][
                distances_to_synthetic[i] <= distances_real_K[i]
            ]
            for i in range(len(distances_to_synthetic))
        ]

        # 3. Count the number of times each synthetic point appears
        ind_synthetic = np.concatenate(ind_synthetic)

        appearance_count = np.zeros(synthetic_data.shape[0])
        bincount = np.bincount(ind_synthetic)
        appearance_count[: len(bincount)] = bincount

        appearance_ratio = appearance_count / self.K
        appearance_ratio = np.clip(appearance_ratio, 0, 1)

        fidelity_ratio = np.mean(appearance_ratio)

        # 4. Compute the ideal value with real data
        # We want to know in how many balls each real point appears
        real_appearance_count = np.bincount(
            ind_real, minlength=self.real_data_neighbors.shape[0]
        )
        real_appearance_ratio = real_appearance_count / self.K
        real_appearance_ratio = np.clip(real_appearance_ratio, 0, 1)

        real_fidelity_ratio = np.mean(real_appearance_ratio)
        fidelity_ratio /= real_fidelity_ratio

        if can_exceed_one:
            return fidelity_ratio
        return fidelity_ratio.clip(max=1)

    def _f_coverage(self, K, N, M, check_code=False):
        # The values are huge, so we use logs to avoid overflow

        results = np.zeros(M - K + 1)

        # compute the log_binomial_coefficient of all 1 <= m <= M+N+1 once
        gamma_ln_all = np.cumsum(np.log(np.arange(1, M + N + 1)))
        gamma_ln_all = np.concatenate(([0], gamma_ln_all))
        if check_code:
            assert int(gamma_ln_all[M]) == int(gammaln(M + 1))

        log_denominator = (
            gamma_ln_all[K - 1] + gamma_ln_all[N - K - 1] - gamma_ln_all[N - 1]
        )
        if check_code:
            assert np.isclose(log_denominator, betaln(K, N - K))

        def log_binomial(n, k):
            return gamma_ln_all[n] - gamma_ln_all[k] - gamma_ln_all[n - k]

        for idx, M_i in enumerate(range(K, M + 1)):
            j_values = np.arange(M_i + 1)
            with np.errstate(divide="ignore"):  # ignore log(0) warnings
                log_min = np.log(np.minimum(j_values, K))

            log_binomial_coeff = log_binomial(M_i, j_values)
            if check_code and M_i == M:
                assert np.isclose(
                    np.log(comb(M, 50, exact=False)), log_binomial_coeff[50]
                )

            log_beta_numerators = (
                gamma_ln_all[K + j_values - 1]
                + gamma_ln_all[N + M_i - K - j_values - 1]
                - gamma_ln_all[N + M_i - 1]
            )
            if check_code and M_i == M:
                assert np.isclose(log_beta_numerators[M], betaln(K + M, N - K))

            log_terms = (
                log_min
                + log_binomial_coeff
                + log_beta_numerators
                - log_denominator
            )
            log_probability = np.logaddexp.reduce(log_terms)

            results[idx] = np.exp(log_probability) / K

        return np.concatenate([np.zeros(K), results])

    def _g(self, y, f_rank_values):
        index = np.searchsorted(f_rank_values, y)
        return index / len(f_rank_values)

    def ClippedCoverage(self, synthetic_data, check_code=False, **kwargs):
        synthetic_data_neighbors = synthetic_data.reshape(
            synthetic_data.shape[0], -1
        )

        if (
            self.f_coverage_values is None
            or synthetic_data_neighbors.shape[0] + 1
            != self.f_coverage_values.shape[0]
        ):
            self.f_coverage_values = self._f_coverage(
                M=synthetic_data_neighbors.shape[0],
                N=self.real_data_neighbors.shape[0],
                K=self.K,
                check_code=check_code,
            )

        if self.distances_real is None:
            neighbors_real = NearestNeighbors(
                n_neighbors=1 + self.K, n_jobs=self.n_jobs
            ).fit(self.real_data_neighbors)
            distances_real, indices_real = neighbors_real.kneighbors(
                self.real_data_neighbors
            )
            self.distances_real = distances_real
            self.indices_real = indices_real

        distances_real_K = self.distances_real[:, -1]  # radius_K

        # 2. For each real point, find the distances to the k closest synthetic
        # data points
        neighbors_synthetic = NearestNeighbors(
            n_neighbors=self.K, n_jobs=self.n_jobs
        ).fit(synthetic_data_neighbors)
        distances_to_synthetic, _ = neighbors_synthetic.kneighbors(
            self.real_data_neighbors
        )

        # 3. For each real point, select the closest synthetic data points that
        # are closer than the real point's k-nearest real neighbor
        mask = distances_to_synthetic < distances_real_K[:, np.newaxis]
        coverage_proportion = np.mean(mask, axis=1)

        # no need to clip, we considered K points
        result_without_g = np.mean(coverage_proportion)
        if kwargs.get(
            "_without_g", False
        ):  # Should never be used, only for fig 3
            return result_without_g
        return self._g(result_without_g, self.f_coverage_values)
