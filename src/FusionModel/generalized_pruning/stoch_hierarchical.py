from .base_gp import BaseGP
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


class StochHierarchical(BaseGP):
    def __init__(self, max_cluster_size=None, alphas=0, weights=None, n_restarts=1000, top_k=5,
                                                n_jobs=-1, verbose=1, memory_efficient=2):
        super().__init__(alphas=alphas, max_cluster_size=max_cluster_size, weights=weights)
        self.n_restarts = n_restarts
        self.top_k = top_k
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.memory_efficient = memory_efficient

    def clustering(self, X, s):
        if self.top_k <= 1:
            self.n_restarts = 1
        if self.memory_efficient == 2:
            D_sq, X_embedded = self._prepare_exact_distances(X)
            seeds = np.random.randint(0, 10000000, size=self.n_restarts)

            def _single_run_score_only(D_sq, X_emb, s, weights, max_cluster_size, top_k, seed):
                full_result = self._stochastic_clustering_pure_distance(
                    D_sq, X_emb, s, weights, max_cluster_size, top_k, seed
                )

                if full_result is None:
                    return None

                inertia = full_result[1]
                return (inertia, seed)

            scores_and_seeds = Parallel(
                n_jobs=self.n_jobs,
                return_as="generator_unordered",
                batch_size=5,
                pre_dispatch='4*n_jobs',
                verbose=0
            )(
                delayed(_single_run_score_only)(
                    D_sq, X_embedded, s, self.weights, self.max_cluster_size, self.top_k, seed
                ) for seed in seeds
            )

            best_inertia = float('inf')
            best_seed = None

            for res in scores_and_seeds:
                if res is not None:
                    inertia, seed = res
                    if inertia < best_inertia:
                        best_inertia = inertia
                        best_seed = seed

            if best_seed is None:
                raise RuntimeError("All restarts failed.")

            best_labels, best_inertia = self._stochastic_clustering_pure_distance(
                D_sq, X_embedded, s, self.weights, self.max_cluster_size, self.top_k, best_seed
            )
            final_centroids = self._reconstruct_centroids(X, best_labels, n_clusters=s)

            best_result = (final_centroids, best_labels, best_inertia)
        elif self.memory_efficient == 1:
            seeds = np.random.randint(0, 10000000, size=self.n_restarts)
            X_embedded, X_original = self._prepare_high_dim_data_high_precision(X)

            def _single_run_score_only(X_embedded, s, weights, max_cluster_size, top_k, seed):
                # Run your logic
                full_result = self._stochastic_clustering_optimized(X_embedded, s, weights, max_cluster_size, top_k, seed)
                if full_result is None:
                    return None
                inertia = full_result[-1]
                return (inertia, seed)

            scores_and_seeds = Parallel(
                n_jobs=self.n_jobs,
                return_as="generator_unordered",
                batch_size=5,  # High batch size is safe now!
                pre_dispatch='4*n_jobs',
                verbose=0
            )(
                delayed(_single_run_score_only)(
                    X_embedded, s, self.weights, self.max_cluster_size, self.top_k, seed
                ) for seed in seeds
            )

            best_inertia = float('inf')
            best_seed = None

            for res in scores_and_seeds:
                if res is not None:
                    inertia, seed = res
                    if inertia < best_inertia:
                        best_inertia = inertia
                        best_seed = seed

            if best_seed is None:
                raise RuntimeError("All restarts failed.")

            labels, inertia = self._stochastic_clustering_optimized(
                X_embedded, s, self.weights, self.max_cluster_size, self.top_k, best_seed
            )
            centroids = self._reconstruct_centroids(X_original, labels, n_clusters=s)
            best_result = (centroids, labels, inertia)
        else:
            seeds = np.random.randint(0, 10000000, size=self.n_restarts)
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._single_stochastic_run)(
                    X, s, self.weights, self.max_cluster_size, self.top_k, seed
                ) for seed in seeds
            )

            valid_results = [r for r in results if r is not None]

            if not valid_results:
                raise RuntimeError("All restarts failed due to constraints.")
            best_result = min(valid_results, key=lambda x: x[2])
        return best_result

    def _stochastic_clustering_optimized(self, X_embedded, s, weights, max_cluster_size, top_k, random_seed):
        """
        Runs the clustering on the N x N embedded data.
        This runs nearly instantly even with multiple repeats.
        """
        np.random.seed(random_seed)
        n_samples, _ = X_embedded.shape  # This is now N x N_components (approx N)

        # --- Standard Initialization (Fast on N x N) ---
        if weights is None:
            W = np.ones(n_samples, dtype=float)
        else:
            W = np.array(weights, dtype=float).ravel()

        active_mask = np.ones(n_samples, dtype=bool)
        cluster_members = [{i} for i in range(n_samples)]
        cluster_sizes = np.ones(n_samples, dtype=int)

        # Pre-compute distances on the embedded data
        # (N x N) operations - trivial compared to N x d
        X_sq = np.sum(X_embedded ** 2, axis=1)
        dist_sq_matrix = X_sq[:, None] + X_sq[None, :] - 2 * (X_embedded @ X_embedded.T)
        dist_sq_matrix = np.maximum(dist_sq_matrix, 0)

        # Initial Cost
        W_matrix = W[:, None]
        weight_factor_matrix = (W_matrix * W_matrix.T) / (W_matrix + W_matrix.T + 1e-20)
        cost_matrix = np.sqrt(2 * dist_sq_matrix * weight_factor_matrix)

        # Masks
        np.fill_diagonal(cost_matrix, np.inf)
        cost_matrix[np.tril(np.ones((n_samples, n_samples), dtype=bool))] = np.inf

        if max_cluster_size is not None:
            size_matrix = cluster_sizes[:, None] + cluster_sizes[None, :]
            cost_matrix[size_matrix > max_cluster_size] = np.inf

        current_centroids = X_embedded.copy()
        num_clusters = n_samples

        # Row Cache (Optimization from previous answer)
        row_min_vals = np.min(cost_matrix, axis=1)
        row_min_idxs = np.argmin(cost_matrix, axis=1)

        while num_clusters > s:
            # --- Selection ---
            valid_rows = np.where(~np.isinf(row_min_vals))[0]
            if len(valid_rows) == 0: return None

            if len(valid_rows) <= top_k:
                candidates = valid_rows
            else:
                partition_idx = np.argpartition(row_min_vals[valid_rows], top_k - 1)[:top_k]
                candidates = valid_rows[partition_idx]

            candidate_costs = row_min_vals[candidates]

            # Softmax
            if len(candidates) == 1:
                chosen_idx = candidates[0]
            else:
                min_c = np.min(candidate_costs)
                scale = np.mean(candidate_costs - min_c)
                if scale < 1e-9:
                    idx_in_cand = np.random.choice(len(candidates))
                else:
                    probs = np.exp(-(candidate_costs - min_c) / scale)
                    probs /= probs.sum()
                    idx_in_cand = np.random.choice(len(candidates), p=probs)
                chosen_idx = candidates[idx_in_cand]

            i = chosen_idx
            j = row_min_idxs[i]
            if i > j: i, j = j, i

            # --- Update ---
            w1, w2 = W[i], W[j]
            new_w = w1 + w2

            # Merge centroids (In embedded space)
            current_centroids[i] = (current_centroids[i] * w1 + current_centroids[j] * w2) / new_w

            cluster_members[i] = cluster_members[i].union(cluster_members[j])
            cluster_members[j] = set()

            new_size = cluster_sizes[i] + cluster_sizes[j]
            cluster_sizes[i] = new_size
            cluster_sizes[j] = 0

            # Lance-Williams Update
            d2_ij = dist_sq_matrix[i, j]
            d2_ik = dist_sq_matrix[i, :]
            d2_jk = dist_sq_matrix[j, :]
            correction = (w1 * w2 * d2_ij) / (new_w ** 2)
            new_dists = np.divide((w1 * d2_ik + w2 * d2_jk), new_w - correction)
            new_dists = np.maximum(new_dists, 0)

            dist_sq_matrix[i, :] = new_dists
            dist_sq_matrix[:, i] = new_dists
            dist_sq_matrix[i, i] = 0.0

            # Cost Update
            W[i] = new_w
            denom = new_w + W
            num = new_w * W
            new_factors = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
            new_costs = np.sqrt(2 * new_dists * new_factors)

            new_costs[~active_mask] = np.inf
            if max_cluster_size is not None:
                new_costs[(new_size + cluster_sizes) > max_cluster_size] = np.inf

            # Matrix updates
            mask_row_i = np.arange(n_samples) > i
            cost_matrix[i, mask_row_i] = new_costs[mask_row_i]
            cost_matrix[i, ~mask_row_i] = np.inf
            cost_matrix[:, i] = np.inf

            # Deactivation
            active_mask[j] = False
            row_min_vals[j] = np.inf
            cost_matrix[j, :] = np.inf
            cost_matrix[:, j] = np.inf

            # Cache Update
            row_min_vals[i] = np.min(cost_matrix[i, :])
            row_min_idxs[i] = np.argmin(cost_matrix[i, :])

            dirty_mask = (row_min_idxs == i) | (row_min_idxs == j)
            dirty_mask[i] = False
            dirty_mask[j] = False
            dirty_mask[~active_mask] = False

            dirty_indices = np.where(dirty_mask)[0]
            if len(dirty_indices) > 0:
                sub_costs = cost_matrix[dirty_indices, :]
                row_min_vals[dirty_indices] = np.min(sub_costs, axis=1)
                row_min_idxs[dirty_indices] = np.argmin(sub_costs, axis=1)

            num_clusters -= 1

        # --- Final K-Means Refinement ---
        # CRITICAL OPTIMIZATION: Run KMeans on the embedded N x N data, NOT the N x d data.
        # This is mathematically equivalent for Euclidean distance.
        active_indices = np.where(active_mask)[0]
        final_centroids_emb = current_centroids[active_indices]

        # Map old labels to 0..s-1
        temp_labels = np.zeros(n_samples, dtype=int)
        mapping = {old: new for new, old in enumerate(active_indices)}
        for old in active_indices:
            for m in cluster_members[old]:
                temp_labels[m] = mapping[old]

        # Initialize KMeans with the result of the hierarchical process
        kmeans = KMeans(n_clusters=s, init=final_centroids_emb, n_init=1, max_iter=1000, tol=1e-5)
        kmeans.fit(X_embedded, sample_weight=weights)

        # Final Output
        final_labels = kmeans.labels_
        final_inertia = kmeans.inertia_

        # Notice: We return labels and inertia here.
        # We do NOT return high-dim centroids yet to save time in the loop.
        return final_labels, final_inertia

    def _reconstruct_centroids(self, X_high_dim, labels, n_clusters):
        """
        Reconstructs the 10^6 dim centroids from the labels.
        This is faster than vector projection because we just average rows.
        """
        n_samples, n_features = X_high_dim.shape
        centroids = np.zeros((n_clusters, n_features))

        # If X is sparse, allow for efficient slicing
        is_sparse = sp.issparse(X_high_dim)

        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) > 0:
                if is_sparse:
                    # Average of sparse rows
                    subset = X_high_dim[members, :]
                    centroids[k] = np.array(subset.mean(axis=0)).flatten()
                else:
                    centroids[k] = np.mean(X_high_dim[members], axis=0)

        return centroids

    def _stochastic_clustering_pure_distance(self, D_sq_original, X_for_kmeans, s, weights, max_cluster_size, top_k,
                                            random_seed):
        """
        Runs clustering purely on the Distance Matrix.
        """
        np.random.seed(random_seed)
        n_samples = D_sq_original.shape[0]

        # --- Initialization ---
        dist_sq_matrix = D_sq_original.copy()

        if weights is None:
            W = np.ones(n_samples, dtype=float)
        else:
            W = np.array(weights, dtype=float).ravel()

        active_mask = np.ones(n_samples, dtype=bool)
        cluster_members = [{i} for i in range(n_samples)]
        cluster_sizes = np.ones(n_samples, dtype=int)

        # Initial Cost Calculation
        W_matrix = W[:, None]
        weight_factor_matrix = (W_matrix * W_matrix.T) / (W_matrix + W_matrix.T + 1e-20)
        cost_matrix = np.sqrt(2 * dist_sq_matrix * weight_factor_matrix)

        # Apply Masks (Upper Triangular)
        np.fill_diagonal(cost_matrix, np.inf)
        cost_matrix[np.tril(np.ones((n_samples, n_samples), dtype=bool))] = np.inf

        if max_cluster_size is not None:
            size_matrix = cluster_sizes[:, None] + cluster_sizes[None, :]
            cost_matrix[size_matrix > max_cluster_size] = np.inf

        # Row Cache
        row_min_vals = np.min(cost_matrix, axis=1)
        row_min_idxs = np.argmin(cost_matrix, axis=1)

        num_clusters = n_samples

        # --- Main Loop ---
        while num_clusters > s:

            # 1. Selection
            valid_rows = np.where(~np.isinf(row_min_vals))[0]
            if len(valid_rows) == 0: return None

            if len(valid_rows) <= top_k:
                candidates = valid_rows
            else:
                # We partition to find the k smallest minimums
                partition_idx = np.argpartition(row_min_vals[valid_rows], top_k - 1)[:top_k]
                candidates = valid_rows[partition_idx]

            candidate_costs = row_min_vals[candidates]

            # Softmax / Greedy Selection
            if len(candidates) == 1:
                chosen_idx = candidates[0]
            else:
                min_c = np.min(candidate_costs)
                scale = np.mean(candidate_costs - min_c)
                if scale < 1e-9:
                    idx_in_cand = np.random.choice(len(candidates))
                else:
                    probs = np.exp(-(candidate_costs - min_c) / scale)
                    probs /= probs.sum()
                    idx_in_cand = np.random.choice(len(candidates), p=probs)
                chosen_idx = candidates[idx_in_cand]

            i = chosen_idx
            j = row_min_idxs[i]

            # Enforce i < j for consistent deletion
            if i > j: i, j = j, i

            # 2. Update Metadata
            w1, w2 = W[i], W[j]
            new_w = w1 + w2

            cluster_members[i] = cluster_members[i].union(cluster_members[j])
            cluster_members[j] = set()

            new_size = cluster_sizes[i] + cluster_sizes[j]
            cluster_sizes[i] = new_size
            cluster_sizes[j] = 0

            # 3. Lance-Williams Distance Update
            d2_ij = dist_sq_matrix[i, j]
            d2_ik = dist_sq_matrix[i, :]
            d2_jk = dist_sq_matrix[j, :]

            correction = (w1 * w2 * d2_ij) / (new_w ** 2)
            new_dists = (w1 * d2_ik + w2 * d2_jk) / new_w - correction
            new_dists = np.maximum(new_dists, 0)

            # Update Distance Matrix (Symmetric)
            dist_sq_matrix[i, :] = new_dists
            dist_sq_matrix[:, i] = new_dists
            dist_sq_matrix[i, i] = 0.0

            # 4. Cost Matrix Update
            W[i] = new_w
            denom = new_w + W
            num = new_w * W

            # Handle division by zero safely
            with np.errstate(divide='ignore', invalid='ignore'):
                new_factors = np.divide(num, denom)
                new_factors[denom == 0] = 0

            new_costs = np.sqrt(2 * new_dists * new_factors)
            new_costs[~active_mask] = np.inf

            if max_cluster_size is not None:
                new_costs[(new_size + cluster_sizes) > max_cluster_size] = np.inf

            # --- FIX STARTS HERE ---

            # A. Update Row i (Upper Triangle: i < k)
            mask_right = (np.arange(n_samples) > i) & active_mask
            cost_matrix[i, mask_right] = new_costs[mask_right]
            cost_matrix[i, ~mask_right] = np.inf

            # B. Update Column i (Upper Triangle: k < i)
            # We must update cost_matrix[k, i] for active k < i
            mask_top = (np.arange(n_samples) < i) & active_mask
            cost_matrix[mask_top, i] = new_costs[mask_top]

            # C. Deactivate j
            active_mask[j] = False
            row_min_vals[j] = np.inf
            cost_matrix[j, :] = np.inf
            cost_matrix[:, j] = np.inf  # Safe to clear col j, it's being removed

            # 5. Cache Update

            # Recompute min for row i
            row_min_vals[i] = np.min(cost_matrix[i, :])
            row_min_idxs[i] = np.argmin(cost_matrix[i, :])

            # Identify rows that need re-scanning:
            # 1. Rows that pointed to i (target value changed)
            # 2. Rows that pointed to j (target died)
            # 3. Rows (k < i) where the new cost to i is BETTER than their previous min

            # Condition 3 check:
            potential_improvements = np.where(mask_top)[0]  # Indices k < i
            if len(potential_improvements) > 0:
                # Check which ones actually improved
                improved_mask = new_costs[potential_improvements] < row_min_vals[potential_improvements]
                improved_indices = potential_improvements[improved_mask]

                # Fast update for these (no need to scan row)
                row_min_vals[improved_indices] = new_costs[improved_indices]
                row_min_idxs[improved_indices] = i

            # Standard dirty check for cases 1 & 2
            # Note: We exclude rows we just "fast updated" to avoid redundant work,
            # though strictly not necessary for correctness.
            dirty_mask = (row_min_idxs == i) | (row_min_idxs == j)
            dirty_mask[i] = False
            dirty_mask[j] = False
            dirty_mask[~active_mask] = False

            dirty_indices = np.where(dirty_mask)[0]

            if len(dirty_indices) > 0:
                sub_costs = cost_matrix[dirty_indices, :]
                row_min_vals[dirty_indices] = np.min(sub_costs, axis=1)
                row_min_idxs[dirty_indices] = np.argmin(sub_costs, axis=1)

            # --- FIX ENDS HERE ---

            num_clusters -= 1

        # --- Final Step: K-Means Refinement ---
        active_indices = np.where(active_mask)[0]
        n_final_clusters = len(active_indices)

        temp_labels = np.zeros(n_samples, dtype=int)
        mapping = {old: new for new, old in enumerate(active_indices)}

        for old_idx in active_indices:
            members = list(cluster_members[old_idx])
            new_idx = mapping[old_idx]
            temp_labels[members] = new_idx

        # Use the K-Means function from the previous step
        # Ensure 'kmeans_from_distance_matrix' is defined in your scope
        _, labels, inertia = self._kmeans_from_distance_matrix(D_sq_original, s, temp_labels, sample_weight=weights)

        return labels, inertia

    def _prepare_high_dim_data_high_precision(self, X):
        """
        Projects High-Dim X (N x d) to (N x N) using Classical MDS
        based on exact pairwise distances.

        This avoids the catastrophic cancellation errors of the Gram Matrix (XX^T) approach.
        """
        n_samples = X.shape[0]

        # --- 1. Exact Distance Calculation ---
        # pdist computes (x-y)^2 directly, which is stable for high-d vectors
        # even if they are far from the origin.
        # We use float64 here as pdist is optimized C code.
        dist_vec = pdist(X, metric='sqeuclidean')
        D_sq = squareform(dist_vec)

        # --- 2. High-Precision Double Centering ---
        # We switch to float128 (if available) or float64 to minimize error
        # during the centering transformation.
        # B = -0.5 * J * D^2 * J  (where J is the centering matrix)

        # Check for extended precision support
        try:
            dtype_prec = np.float128
        except AttributeError:
            dtype_prec = np.float64

        D_sq = D_sq.astype(dtype_prec)

        # Calculate row/col means with high precision
        row_means = np.mean(D_sq, axis=1, keepdims=True)
        col_means = np.mean(D_sq, axis=0, keepdims=True)
        grand_mean = np.mean(D_sq)

        # Apply Double Centering Formula: B_ij = -0.5 * (D^2_ij - row_mean - col_mean + grand_mean)
        B = -0.5 * (D_sq - row_means - col_means + grand_mean)

        # --- 3. Eigen Decomposition ---
        # Recover coordinates from the centered Gram matrix B
        # B is symmetric, so we use eigh
        eigvals, eigvecs = eigh(B)

        # --- 4. Filtering and Reconstruction ---
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Clip negative eigenvalues (numerical noise) to zero
        eigvals[eigvals < 0] = 0

        # Recover coordinates: X = E * sqrt(Lambda)
        # We cast back to float64 for the actual clustering loop speed
        eigvals_sqrt = np.sqrt(eigvals).astype(np.float64)
        eigvecs = eigvecs.astype(np.float64)

        X_embedded = eigvecs * eigvals_sqrt

        return X_embedded, X

    def _prepare_exact_distances(self, X):
        dist_vec = pdist(X, metric='sqeuclidean')
        D_sq = squareform(dist_vec)
        return D_sq, X