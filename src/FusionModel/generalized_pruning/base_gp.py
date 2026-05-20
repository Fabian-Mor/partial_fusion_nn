import numpy as np
import warnings
from sklearn.cluster import KMeans

from src.CNN import VGG11
from src.FusionModel.fusion_methods import PartialFusion
from src.CNN.VGG import _reconstruct_network
from src.FusionModel.fusion_model import FusionModel

class BaseGP:
    def __init__(self, alphas=None, max_cluster_size=None, weights=None):
        self.max_cluster_size = max_cluster_size
        self.weights = weights
        if alphas is None:
            self.alphas = [0]
        elif not isinstance(alphas, list):
            self.alphas = [alphas]
        else:
            self.alphas = alphas
        self.A1_list = None
        self.A2_list = None
        self.A3_list = None

    def combine_layers(self, model1, model2, data, lambdas, layer):
        if self.A1_list is None or self.A2_list is None or self.A3_list is None:
            A1, A2, A3 = self._fuse_clustering(model1, model2, data=data, lambdas=lambdas)
            self.A1_list = A1
            self.A2_list = A2
            self.A3_list = A3
        return self.A1_list[layer], self.A2_list[layer], self.A3_list[layer]

    def prune_network(self, model, model_size, mult, model_cls, data, which_act=0):
        if model_cls is VGG11:
            return self._prune_cnn(model, model_size, mult, data)
        else:
            return self._prune_ff(model, model_size, mult, model_cls, data, which_act=which_act)

    def _prune_ff(self, model, model_size, mult, model_cls, data, which_act=0):
        smaller_sizes = [int(np.round(mult * l)) for l in model_size]
        layers1 = model.get_layer_names_with_weights()
        L = len(layers1)
        act_list = []
        quant_list = []
        pi_list = []
        for l in range(L - 1):  # only until L-2 as L-1 is just the output layer, where we have no compression and just the identity kernel
            layer_name = layers1[l]
            if data is not None:
                activations = model.get_activations(layer_name, data, numpy=True).T  # activations pre-activaiton function; but activation is applied elow with func
                activations /= np.linalg.norm(activations)
            else:
                activations = model.get_incoming_weights(layer_name, numpy=True)
                activations /= np.linalg.norm(activations)
            len_act, _ = activations.shape
            func = model.get_next_activation(layer_name, numpy=True)
            if (func is not None) and (data is not None):
                activations = func(activations)

            act_list.append(activations)
            centroids, labels, inertia = self.clustering(activations, smaller_sizes[l])

            weights = np.array([np.sum(labels == i) for i in range(smaller_sizes[l])]) / len_act
            quant_list.append([centroids, weights])

            # Below: calculate pi according to clustering
            pi = np.zeros([len_act, smaller_sizes[l]])
            for ind_cl in range(int(smaller_sizes[l])):
                pi[:, ind_cl] = (labels == ind_cl) / len_act
            pi_list.append(pi)

        input_length = model.get_incoming_weights(layers1[0], numpy=True).shape[1]
        kernels_forward = [np.identity(input_length)]
        kernels_backward = [np.identity(input_length)]
        for l in range(L):
            layer_name = layers1[l]
            acti_for_len = model.get_activations(layer_name, data, numpy=True).T
            len_act, _ = acti_for_len.shape
            if l == L - 1:
                kernels_forward.append(np.identity(len_act))
                kernels_backward.append(np.identity(len_act))
            else:
                pi = pi_list[l]
                #print(pi.shape)
                w_mu = np.sum(pi, axis=1)
                w_quant = np.sum(pi, axis=0) + 1e-15

                k_for = np.transpose(pi / w_mu[:, None])  # kernel forward resulting from pi, so pi = mu times k_for
                k_back = np.divide(pi, w_quant[None, :])  # kernel backward resulting from pi, so pi = nu times k_back
                kernels_forward.append(k_for.copy())
                kernels_backward.append(k_back.copy())

        weight_list = []
        for l in range(L):
            k_back_incoming = kernels_backward[l]
            k_for_outgoing = kernels_forward[l + 1]
            w_a = model.get_incoming_weights(layers1[l], numpy=True)
            w_b = k_for_outgoing @ w_a @ k_back_incoming
            weight_list.append(w_b.copy())
        out_net = model_cls(hidden_size_1=smaller_sizes[0], hidden_size_2=smaller_sizes[1],
                           hidden_size_3=smaller_sizes[2], assign_w=True, w_mats=weight_list, which_act=which_act)
        return out_net

    def _prune_cnn(self, model, model_size, mult, data):
        smaller_sizes = [int(np.round(mult * l)) for l in model_size]
        layers = model.get_layer_names_with_weights()
        layers = [l for l in layers if len(model.get_incoming_weights(l, numpy=True).shape) == 4]

        pi_list = []

        for l, layer_name in enumerate(layers):
            n_keep = smaller_sizes[l]
            acts = model.get_activations(layer_name, data, numpy=True)
            acts = np.transpose(acts, (1, 0, 2, 3))
            features = acts.reshape(acts.shape[0], -1)
            n_channels = features.shape[0]
            centroids, labels, inertia = self.clustering(features, smaller_sizes[l])

            #print('pre inertia', inertia)
            if centroids is not None:
                kmeans = KMeans(n_clusters=n_keep, init=centroids, max_iter=300, tol=1e-4)
                kmeans.fit(features)
                labels = kmeans.labels_
                inertia = kmeans.inertia_
            #print('inertia final', inertia)
            pi = np.zeros((n_channels, n_keep))
            for k in range(n_keep):
                indices = np.where(labels == k)[0]
                if len(indices) > 0:
                    pi[indices, k] = 1.0 / n_channels  # Uniform weight distribution
            pi_list.append(pi)
        return _reconstruct_network(model, layers, pi_list, smaller_sizes)


    def _fuse_clustering(self, model1, model2, data=None, lambdas=None):
        net = FusionModel(model1, model2, PartialFusion(alphas=1), lambdas=lambdas)
        pi_list = self._compute_pis(net, data, lambdas)

        layers1 = net.get_layer_names_with_weights()
        L = len(layers1)
        input_length = net.get_incoming_weights(layers1[0], numpy=True).shape[1]
        kernels_forward = [np.identity(input_length)]
        kernels_backward = [np.identity(input_length)]
        for l in range(L):
            layer_name = layers1[l]
            acti_for_len = net.get_activations(layer_name, data, numpy=True).T
            len_act = acti_for_len.shape[0]
            if l == L - 1:
                kernels_forward.append(np.identity(len_act))
                kernels_backward.append(np.identity(len_act))
            else:
                pi = pi_list[l]
                w_mu = np.sum(pi, axis=1)
                w_quant = np.sum(pi, axis=0) + 10 ** -15

                k_for = np.transpose(pi / w_mu[:, None])  # kernel forward resulting from pi, so pi = mu times k_for
                k_back = np.divide(pi, w_quant[None, :])  # kernel backward resulting from pi, so pi = nu times k_back
                kernels_forward.append(k_for.copy())
                kernels_backward.append(k_back.copy())

        weight_list = []
        for l in range(L):
            k_back_incoming = kernels_backward[l]
            k_for_outgoing = kernels_forward[l + 1]
            w_a = net.get_incoming_weights(layers1[l], numpy=True)
            if len(w_a.shape) == 4:
                tmp = np.einsum('bchw,cd->bdhw', w_a, k_back_incoming)
                w_b = np.tensordot(k_for_outgoing, tmp, axes=([1], [0]))
            else:
                w_b = k_for_outgoing @ w_a @ k_back_incoming
            weight_list.append(w_b.copy())

        A1 = {}
        A2 = {}
        A3 = {}
        for i, layer in enumerate(model1.get_layer_names_with_weights()):
            A1[layer] = weight_list[i]
            if len(weight_list[i].shape) == 4:
                A2[layer] = np.zeros((weight_list[i].shape[0], 0, weight_list[i].shape[2], weight_list[i].shape[3]))
                A3[layer] = np.zeros((weight_list[i].shape[0], 0, weight_list[i].shape[2], weight_list[i].shape[3]))
            else:
                A2[layer] = np.zeros((weight_list[i].shape[0], 0))
                A3[layer] = np.zeros((weight_list[i].shape[0], 0))
        return A1, A2, A3

    def _compute_pis(self, net, data, lambdas):
        layers1 = net.get_layer_names_with_weights()
        L = len(layers1)
        pi_list = []
        for l in range(L - 1):
            layer_name = layers1[l]
            activations = net.get_activations(layer_name, data, numpy=True)
            if len(activations.shape) == 2:
                activations = activations.T
            elif len(activations.shape) == 4:
                activations = activations.transpose(1, 0, 2, 3).reshape(activations.shape[1], -1)
            activations /= np.linalg.norm(activations)
            len_act = activations.shape[0]
            func = net.get_next_activation(layer_name, numpy=True)
            if func is not None:
                activations = func(activations)
            alpha = self.alphas[l % len(self.alphas)]
            smaller_size = int((activations.shape[0] / 2) * (1 + alpha))
            if lambdas is not None:
                self.weights = np.ones(activations.shape[0])
                split_point = int(activations.shape[0] / 2)
                self.weights[:split_point] = lambdas[0] + 10**-15
                self.weights[split_point:] = lambdas[1] + 10**-15
            centroids, labels, inertia = self.clustering(activations, smaller_size)

            pi = np.zeros([len_act, smaller_size])
            for ind_cl in range(int(smaller_size)):
                pi[:, ind_cl] = (labels == ind_cl) / len_act
            pi_list.append(pi)
        return pi_list


    def clustering(self, X, s):
        # should return centroids, labels, inertia
        raise NotImplementedError


    def _single_stochastic_run(self, X, s, weights, max_cluster_size, top_k, random_seed):
        """
        A single run of the stochastic clustering, isolated for parallel execution.
        """
        np.random.seed(random_seed)

        n_samples = X.shape[0]

        if weights is None:
            current_weights = np.ones((n_samples, 1))
        else:
            current_weights = np.array(weights, dtype=float).reshape(-1, 1)

        current_centroids = X.copy()
        cluster_members = [{i} for i in range(n_samples)]
        num_clusters = n_samples

        while num_clusters > s:
            C = current_centroids
            W = current_weights

            # 1. Cost Calculation
            dist_sq_matrix = np.sum(C ** 2, axis=1, keepdims=True) + \
                             np.sum(C ** 2, axis=1) - 2 * (C @ C.T)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                weight_factor_matrix = (W * W.T) / (W + W.T)

            cost_matrix = np.sqrt(np.maximum(2 * dist_sq_matrix * weight_factor_matrix, 0))

            # 2. Constraints
            np.fill_diagonal(cost_matrix, np.inf)
            mask = np.tril(np.ones_like(cost_matrix, dtype=bool))
            cost_matrix[mask] = np.inf

            if max_cluster_size is not None:
                sizes = np.array([len(m) for m in cluster_members])
                combined_sizes = sizes[:, None] + sizes[None, :]
                cost_matrix[combined_sizes > max_cluster_size] = np.inf

            # 3. Guided Stochastic Selection
            flat_costs = cost_matrix.ravel()
            valid_merges_count = np.sum(~np.isinf(flat_costs))

            if valid_merges_count == 0:
                return None  # Deadlock

            current_k = min(top_k, valid_merges_count)

            if current_k == 1:
                chosen_flat_idx = np.argmin(flat_costs)
            else:
                partitioned_indices = np.argpartition(flat_costs, current_k - 1)[:current_k]
                candidate_costs = flat_costs[partitioned_indices]

                # Softmax selection
                min_c = np.min(candidate_costs)
                relative_costs = candidate_costs - min_c
                scale = np.mean(relative_costs)

                if scale < 1e-9:
                    probs = np.ones(current_k) / current_k
                else:
                    weights_prob = np.exp(-relative_costs / scale)
                    probs = weights_prob / np.sum(weights_prob)

                chosen_flat_idx = np.random.choice(partitioned_indices, p=probs)

            i, j = np.unravel_index(chosen_flat_idx, cost_matrix.shape)
            if i > j: i, j = j, i

            # 4. Update
            c1, c2 = current_centroids[i], current_centroids[j]
            w1, w2 = current_weights[i, 0], current_weights[j, 0]
            new_w = w1 + w2
            new_c = (c1 * w1 + c2 * w2) / new_w
            new_members = cluster_members[i].union(cluster_members[j])

            current_centroids = np.delete(current_centroids, j, axis=0)
            current_weights = np.delete(current_weights, j, axis=0)
            cluster_members.pop(j)

            current_centroids[i] = new_c
            current_weights[i] = new_w
            cluster_members[i] = new_members

            num_clusters -= 1

        # --- Final Evaluation ---
        final_labels = np.zeros(n_samples, dtype=int)
        for c_idx, members in enumerate(cluster_members):
            for s_idx in members:
                final_labels[s_idx] = c_idx

        full_w = np.ones(n_samples) if weights is None else np.array(weights)
        assigned_c = current_centroids[final_labels]
        sq_dists = np.sum((X - assigned_c) ** 2, axis=1)
        total_inertia = np.sum(full_w * sq_dists)

        kmeans = KMeans(n_clusters=s, init=current_centroids, max_iter=10 ** 4, tol=10 ** -5).fit(X, sample_weight=weights)
        current_centroids, final_labels, total_inertia = kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_

        return (current_centroids, final_labels, total_inertia)

    def _kmeans_from_distance_matrix(self, D_sq, n_clusters, init_labels, sample_weight=None, max_iter=10000, tol=1e-5):
        """
        Performs K-Means clustering using a pairwise distance matrix.

        Parameters:
        - D: (n_samples, n_samples) Symmetric matrix of Euclidean distances.
        - n_clusters: int, number of clusters.
        - init_labels: (n_samples,) array of initial integer labels (0 to n_clusters-1).
        - sample_weight: (n_samples,) array of weights (optional).
        - max_iter: int, maximum number of iterations.
        - tol: float, tolerance for convergence (relative change in inertia).

        Returns:
        - cluster_centers_: None (Cannot be computed explicitly without coordinates).
        - labels_: (n_samples,) Predicted labels.
        - inertia_: float, Sum of squared distances of samples to their closest cluster center.
        """
        n_samples = D_sq.shape[0]

        # Handle weights
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)

        # Initialize
        labels = np.array(init_labels, dtype=int)
        inertia = np.inf

        # Pre-allocate distance matrix (Samples x Clusters)
        dist_to_centroids = np.zeros((n_samples, n_clusters))

        for iteration in range(max_iter):
            prev_labels = labels.copy()
            prev_inertia = inertia

            # --- Step 1: Update Implicit Centroids & Calculate Distances ---
            for k in range(n_clusters):
                # Indices of points currently in cluster k
                mask = (labels == k)

                # Handle empty clusters (standard KMeans strategy: skip or re-init; here we skip)
                if not np.any(mask):
                    dist_to_centroids[:, k] = np.inf
                    continue

                w_k = sample_weight[mask]
                W_k = np.sum(w_k)  # Total weight of cluster

                # Term 1: Weighted average squared distance from every point i to points in C_k
                # Shape: (n_samples, subset) @ (subset,) -> (n_samples,)
                term1 = np.divide((D_sq[:, mask] @ w_k), (W_k))

                # Term 2: Weighted internal variance of the cluster (constant for the cluster)
                # Shape: (subset,) @ (subset, subset) @ (subset,) -> scalar
                # We subset D_sq to only the rows/cols belonging to the cluster
                D_sq_subset = D_sq[np.ix_(mask, mask)]
                term2 = np.divide((w_k @ D_sq_subset @ w_k), (2 * (W_k ** 2)))

                # Combine to get squared distance to the implicit centroid
                dist_to_centroids[:, k] = term1 - term2

            # --- Step 2: Assignment ---
            # Clip negative values that might occur due to floating point errors
            dist_to_centroids = np.maximum(dist_to_centroids, 0)

            labels = np.argmin(dist_to_centroids, axis=1)

            # --- Step 3: Calculate Inertia ---
            # Inertia is sum of squared distances to the nearest centroid
            min_dists = np.min(dist_to_centroids, axis=1)
            inertia = np.sum(sample_weight * min_dists)

            # --- Step 4: Convergence Check ---
            # Check label stability
            if np.array_equal(labels, prev_labels):
                break

            # Check inertia tolerance
            if prev_inertia != np.inf:
                change = np.abs(prev_inertia - inertia)
                if change < tol:
                    break

        # Cluster centers cannot be returned as coordinates (F dims) because input was only D (N dims).
        # We return None to maintain the 3-element tuple structure requested.
        cluster_centers_ = None

        return cluster_centers_, labels, inertia