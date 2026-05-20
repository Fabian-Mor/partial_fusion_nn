from .base_gp import BaseGP
import numpy as np
import warnings


class WeightHierarchical(BaseGP):
    def __init__(self, alphas=0, max_cluster_size=None, weights=None, weight_add=0):
        super().__init__(alphas=alphas, max_cluster_size=max_cluster_size, weights=weights)
        self.weight_add = weight_add

    def clustering(self, X, s):
        """
        Performs a fast, weight-aware hierarchical clustering that is designed to be
        compatible with SciPy's 'ward' method in the unweighted case.

        The cost function is modified to match SciPy's linkage criterion, which is
        sqrt(2 * increase_in_SSE).

        Args:
            X (np.ndarray): The input data matrix of size [n, d].
            s (int): The desired final number of clusters.
            weights (np.ndarray, optional): A vector of weights of size [n].
                                            If None, uniform weights of 1 are used.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: A matrix of weighted cluster midpoints of size [s, d].
                - np.ndarray: A vector of labels of size [n].
                - float: The final weighted inertia.
        """
        n_samples, n_features = X.shape

        # --- Step 1: Initialization ---
        if self.weights is None:
            current_weights = np.ones((n_samples, 1))
        else:
            current_weights = np.array(self.weights, dtype=float).reshape(-1, 1)
            if current_weights.shape[0] != n_samples:
                raise ValueError("Shape of weights must match the number of samples in X.")

        current_centroids = X.copy()
        cluster_members = [{i} for i in range(n_samples)]
        num_clusters = n_samples

        # --- Step 2: Vectorized Iterative Merging ---
        while num_clusters > s:
            k = num_clusters

            C = current_centroids
            W = current_weights

            # --- Vectorized cost calculation ---
            dist_sq_matrix = np.sum(C ** 2, axis=1, keepdims=True) + np.sum(C ** 2, axis=1) - 2 * (C @ C.T)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                weight_factor_matrix = (W * W.T) / (W + W.T)

            increase_in_sse_matrix = dist_sq_matrix * weight_factor_matrix

            # --- KEY CHANGE HERE ---
            # Match SciPy's linkage distance: sqrt(2 * increase_in_SSE)
            # We handle potential negative values from floating point errors.
            cost_matrix = np.sqrt(np.maximum(2 * increase_in_sse_matrix, 0))

            bias = np.array([len(cluster_members[i]) for i in range(len(cost_matrix))])
            cost_bias = (bias[:, None] + bias[None, :]) * self.weight_add

            np.fill_diagonal(cost_matrix, np.inf)

            # --- Find and perform the best merge ---
            i, j = np.unravel_index(np.argmin(cost_matrix + cost_bias), cost_matrix.shape)
            if i > j: i, j = j, i

            c1, c2 = current_centroids[i], current_centroids[j]
            w1, w2 = current_weights[i, 0], current_weights[j, 0]

            new_weight = w1 + w2
            new_centroid = (c1 * w1 + c2 * w2) / new_weight
            new_members = cluster_members[i].union(cluster_members[j])

            current_centroids = np.delete(current_centroids, j, axis=0)
            current_weights = np.delete(current_weights, j, axis=0)
            cluster_members.pop(j)

            current_centroids[i] = new_centroid
            current_weights[i] = new_weight
            cluster_members[i] = new_members

            num_clusters -= 1

        # --- Step 4: Finalize and return results ---
        final_centroids = current_centroids
        final_labels = np.zeros(n_samples, dtype=int)
        for cluster_idx, members in enumerate(cluster_members):
            for sample_idx in members:
                final_labels[sample_idx] = cluster_idx

        full_weights = np.ones(n_samples) if self.weights is None else np.array(self.weights)
        assigned_centroids = final_centroids[final_labels]
        squared_distances = np.sum((X - assigned_centroids) ** 2, axis=1)
        final_inertia = np.sum(full_weights * squared_distances)

        return final_centroids, final_labels, final_inertia