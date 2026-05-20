import numpy as np
import ot
from sklearn.cluster import KMeans
import sys
sys.path.append("../../../")
from src.FusionModel.generalized_pruning import WeightHierarchical, StochHierarchical
from src.FusionModel import FusionModel
from src.FusionModel.fusion_methods.naive_fusion import NaiveFusion
from src.CNN import VGG11

def _get_layer_features(net, layer_name, t_dat, weight_based=False, normalize=False):
    """Extracts features for a layer (either activations or weight vectors)."""
    if weight_based:
        # Use incoming weights as features (normalized)
        w = net.get_incoming_weights(layer_name, numpy=True)
        # Flatten: (Out, In, H, W) -> (Out, Features)
        features = w.reshape(w.shape[0], -1)
    else:
        # Use activations from training data
        acts = net.get_activations(layer_name, t_dat, numpy=True)
        # Flatten: (Batch, Channels, H, W) -> (Channels, Features)
        # We transpose to match channel-first indexing for clustering
        acts = np.transpose(acts, (1, 0, 2, 3))
        features = acts.reshape(acts.shape[0], -1)

    # Normalize features
    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = np.divide(features, norms, out=np.zeros_like(features), where=norms != 0)
    return features

def _reconstruct_network(net, layers, pi_list, smaller_sizes, fusion_network=False):
    weight_list = []

    # Input dim of the network
    input_dim = net.get_incoming_weights(layers[0], numpy=True).shape[1]
    k_back_prev = np.eye(input_dim)

    for l, layer_name in enumerate(layers):
        w_old = net.get_incoming_weights(layer_name, numpy=True)
        pi = pi_list[l]

        w_mu = np.sum(pi, axis=1)
        w_nu = np.sum(pi, axis=0)

        k_for = np.transpose(np.divide(pi, w_mu[:, None], out=np.zeros_like(pi), where=w_mu[:, None] != 0))
        k_back = np.divide(pi, w_nu[None, :], out=np.zeros_like(pi), where=w_nu[None, :] != 0)

        if len(w_old.shape) == 2:
            w_new = k_for @ w_old @ k_back_prev
        else:
            tmp = np.einsum('bchw,cd->bdhw', w_old, k_back_prev)
            w_new = np.tensordot(k_for, tmp, axes=([1], [0]))

        weight_list.append(w_new)
        k_back_prev = k_back

        # --- Build New Net ---
    out_net = VGG11(manual_chanel_sizes=smaller_sizes, neurons_classifier=smaller_sizes[-1])
    if fusion_network:
        out_net = FusionModel(out_net, out_net, NaiveFusion(), lambdas=[1, 0])
    # Set weights for all processed layers
    for l, layer_name in enumerate(layers):
        out_net.set_incoming_weights(layer_name, weight_list[l], numpy=True)

    # --- Final Layer Safety Check ---
    # If the network has a final linear layer ('linear') that was NOT in 'layers',
    # we must adapt its input. If it WAS in 'layers', it's already done.
    if fusion_network:
        last_layer_name = 'fused_layers.22'
    else:
        last_layer_name = 'linear'

    if last_layer_name not in layers:
        print('last layer was not pruned')
        w_last = net.get_incoming_weights(last_layer_name, numpy=True)
        # Handle Conv->Linear or Linear->Linear transition
        if w_last.shape[1] != k_back_prev.shape[0]:
            # Conv -> Linear
            n_channels_old = k_back_prev.shape[0]
            flattened_dim = w_last.shape[1]
            spatial_area = flattened_dim // n_channels_old

            w_last_reshaped = w_last.reshape(w_last.shape[0], n_channels_old, spatial_area)
            w_last_contracted = np.einsum('ocs,cn->ons', w_last_reshaped, k_back_prev)
            w_last_new = w_last_contracted.reshape(w_last.shape[0], -1)
        else:
            # Linear -> Linear
            w_last_new = np.einsum('oi,id->od', w_last, k_back_prev)

        out_net.set_incoming_weights(last_layer_name, w_last_new, numpy=True)

    return out_net

def prune_simple(net, smaller_sizes, only_conv=True, importance_scaling=None, importance_type='incoming', fusion_network=False):
    """
    Structured pruning based on weight norms (L2).
    Now supports pruning the last convolutional layer.
    """
    all_layers = net.get_layer_names_with_weights()
    target_layers = [l for l in all_layers if
                     len(net.get_incoming_weights(l, numpy=True).shape) == 4] if only_conv else all_layers

    pi_list = []

    for l, layer_name in enumerate(target_layers):
        w_curr = net.get_incoming_weights(layer_name, numpy=True)
        n_channels = w_curr.shape[0]

        # Case 1: No target size provided for this layer (keep original)
        if l >= len(smaller_sizes):
            pi = np.eye(n_channels)
            pi_list.append(pi)
            continue

        # Case 2: Pruning requested
        n_keep = smaller_sizes[l]

        # --- Calculate Importance ---
        if importance_type == 'incoming':
            # Norm of the filter: (Out, In, H, W) -> (Out,)
            importance = np.linalg.norm(w_curr.reshape(n_channels, -1), axis=1)

        elif importance_type == 'outgoing':
            # Determine next layer weights for importance
            curr_idx_in_all = all_layers.index(layer_name)

            if curr_idx_in_all + 1 < len(all_layers):
                next_layer = all_layers[curr_idx_in_all + 1]
                w_next = net.get_incoming_weights(next_layer, numpy=True)

                if len(w_next.shape) == 4:
                    # Conv: (Next, Curr, H, W)
                    w_next_re = np.transpose(w_next, (1, 0, 2, 3))
                    importance = np.linalg.norm(w_next_re.reshape(n_channels, -1), axis=1)
                else:
                    # Linear: (Next, Curr * H * W)
                    # We must reshape to isolate the Channel dim
                    pixels = w_next.shape[1] // n_channels
                    # Reshape to (Next, Curr, Pixels)
                    w_next_re = w_next.reshape(w_next.shape[0], n_channels, pixels)
                    # Importance is norm over Next(0) and Pixels(2)
                    # Transpose to (Curr, Next, Pixels)
                    w_next_re = np.transpose(w_next_re, (1, 0, 2))
                    importance = np.linalg.norm(w_next_re.reshape(n_channels, -1), axis=1)
            else:
                importance = np.linalg.norm(w_curr.reshape(n_channels, -1), axis=1)

        # --- Selection ---
        if importance_scaling is not None:
            importance_scale = importance_scaling[l]
            importance *= importance_scale
        # Get indices of top N channels
        top_indices = np.argsort(importance)[::-1][:n_keep]

        pi = np.zeros((n_channels, n_keep))
        for new_idx, old_idx in enumerate(top_indices):
            pi[old_idx, new_idx] = 1.0

        pi_list.append(pi)

    return _reconstruct_network(net, target_layers, pi_list, smaller_sizes, fusion_network=fusion_network)

def _reconstruct_fusion(net, layers, pi_list, keep_indices_list, smaller_sizes, alpha=0.5, fusion_network=False):
    """
    Reconstructs the network by averaging the OT-aligned weights and the
    standard pruned weights (Paper's method).

    w_final = alpha * w_pruned + (1-alpha) * w_aligned
    """
    weight_list = []

    # Track inputs for the pruned path
    # First layer input is full image, so "indices" are just all indices
    input_dim = net.get_incoming_weights(layers[0], numpy=True).shape[1]
    prev_indices_pruned = np.arange(input_dim)

    # Track kernels for the aligned path
    k_back_prev_aligned = np.eye(input_dim)

    for l, layer_name in enumerate(layers):
        w_old = net.get_incoming_weights(layer_name, numpy=True)

        # --- Path A: Aligned (OT) ---
        pi = pi_list[l]
        w_mu = np.sum(pi, axis=1)
        w_nu = np.sum(pi, axis=0)

        # OT Kernels
        k_for = np.transpose(np.divide(pi, w_mu[:, None], out=np.zeros_like(pi), where=w_mu[:, None] != 0))
        k_back = np.divide(pi, w_nu[None, :], out=np.zeros_like(pi), where=w_nu[None, :] != 0)

        if len(w_old.shape) == 2:
            w_aligned = k_for @ w_old @ k_back_prev_aligned
        else:
            tmp = np.einsum('bchw,cd->bdhw', w_old, k_back_prev_aligned)
            w_aligned = np.tensordot(k_for, tmp, axes=([1], [0]))

        k_back_prev_aligned = k_back  # Update for next layer

        # --- Path B: Pruned (Selection) ---
        current_indices_pruned = keep_indices_list[l]

        if len(w_old.shape) == 2:
            # Linear Layer: Select rows (current) and cols (prev)
            # Note: This logic assumes Linear->Linear.
            # If Conv->Linear, we handle it below in the special block.
            w_pruned = w_old[current_indices_pruned, :][:, prev_indices_pruned]
        else:
            # Conv Layer: Select filters (current) and channels (prev)
            # w_old is (Out, In, H, W)
            # Select Out
            w_subset = w_old[current_indices_pruned, :, :, :]
            # Select In
            w_pruned = w_subset[:, prev_indices_pruned, :, :]

        prev_indices_pruned = current_indices_pruned  # Update for next layer

        # --- Fusion ---
        w_final = alpha * w_pruned + (1 - alpha) * w_aligned
        weight_list.append(w_final)

    # --- Final Linear Layer Handling ---
    if fusion_network:
        last_layer_name = 'fused_layers.22'
    else:
        last_layer_name = 'linear'
    w_last = net.get_incoming_weights(last_layer_name, numpy=True)

    # 1. Aligned Path (OT) for Linear Layer
    if w_last.shape[1] != k_back_prev_aligned.shape[0]:
        # Handle Conv->Linear Flattening
        n_channels_old = k_back_prev_aligned.shape[0]
        flattened_dim = w_last.shape[1]
        spatial_area = flattened_dim // n_channels_old

        w_last_reshaped = w_last.reshape(w_last.shape[0], n_channels_old, spatial_area)
        w_last_contracted = np.einsum('ocs,cn->ons', w_last_reshaped, k_back_prev_aligned)
        w_last_aligned = w_last_contracted.reshape(w_last.shape[0], -1)
    else:
        w_last_aligned = np.einsum('oi,id->od', w_last, k_back_prev_aligned)

    # 2. Pruned Path (Selection) for Linear Layer
    # We need to slice the input features corresponding to the kept Conv channels
    if w_last.shape[1] != len(prev_indices_pruned):
        # Conv->Linear: Input is (Channels * H * W)
        n_channels_old = w_last.shape[1] // (w_last.shape[1] // k_back_prev_aligned.shape[0])  # infer from old
        spatial_area = w_last.shape[1] // n_channels_old

        # Reshape to (Out, Channels, Area)
        w_last_reshaped = w_last.reshape(w_last.shape[0], n_channels_old, spatial_area)
        # Select kept channels
        w_last_subset = w_last_reshaped[:, prev_indices_pruned, :]
        w_last_pruned = w_last_subset.reshape(w_last.shape[0], -1)
    else:
        # Linear->Linear
        w_last_pruned = w_last[:, prev_indices_pruned]

    # 3. Final Fusion
    w_last_final = alpha * w_last_pruned + (1 - alpha) * w_last_aligned

    # --- Build Net ---
    out_net = VGG11(manual_chanel_sizes=smaller_sizes, neurons_classifier=smaller_sizes[-1])
    if fusion_network:
        out_net = FusionModel(out_net, out_net, NaiveFusion(), lambdas=[1, 0])
    for l, layer_name in enumerate(layers):
        out_net.set_incoming_weights(layer_name, weight_list[l], numpy=True)
    out_net.set_incoming_weights(last_layer_name, w_last_final, numpy=True)

    return out_net

def prune_ot_paper(net, smaller_sizes, t_dat, alpha=0.5, only_conv=True, normalize=True, impol1=False, fusion_network=False):
    all_layers = net.get_layer_names_with_weights()
    target_layers = [l for l in all_layers if
                     len(net.get_incoming_weights(l, numpy=True).shape) == 4] if only_conv else all_layers

    pi_list = []
    keep_indices_list = []

    for l, layer_name in enumerate(target_layers):
        # ... (Selection logic remains the same) ...
        n_keep = smaller_sizes[l]
        w_curr = net.get_incoming_weights(layer_name, numpy=True)
        n_channels = w_curr.shape[0]

        # Determine surviving indices (L1 or L2 pruning)
        if impol1:
            importance = np.linalg.norm(w_curr.reshape(n_channels, -1), axis=1, ord=1)
            keep_indices = np.argsort(importance)[::-1][:n_keep]
            keep_indices_list.append(keep_indices)
        else:
            importance = np.linalg.norm(w_curr.reshape(n_channels, -1), axis=1)
            keep_indices = np.argsort(importance)[::-1][:n_keep]
            keep_indices_list.append(keep_indices)


        X_full = w_curr.reshape(n_channels, -1)

        if normalize:
            norms = np.linalg.norm(X_full, axis=1, keepdims=True)
            X_full = np.divide(X_full, norms+1e-9, out=np.zeros_like(X_full), where=norms != 0)

        X_centers = X_full[keep_indices]

        # Metric: Squared Euclidean on raw weights
        M = ot.dist(X_full, X_centers, metric='sqeuclidean')

        M = M / (M.max() + 1e-9)

        w_source = np.ones(n_channels) / n_channels
        w_target = np.ones(n_keep) / n_keep

        pi = ot.emd(w_source, w_target, M)
        pi_list.append(pi)

    return _reconstruct_fusion(net, target_layers, pi_list, keep_indices_list, smaller_sizes, alpha=alpha, fusion_network=fusion_network)

def prune_clustering(net, smaller_sizes, t_dat, importance_scaling=None, weight_based=False, clustering_algo='stoch', max_size=None, fusion_network=False):
    layers = net.get_layer_names_with_weights()
    layers = [l for l in layers if len(net.get_incoming_weights(l, numpy=True).shape) == 4]

    pi_list = []

    for l, layer_name in enumerate(layers):
        n_keep = smaller_sizes[l]

        # 1. Get Features (Activations work best for clustering usually)
        features = _get_layer_features(net, layer_name, t_dat, weight_based)
        n_channels = features.shape[0]
        if importance_scaling is not None:
            importance_scale = importance_scaling[l]
        else:
            importance_scale = None
        # 2. Run K-Means
        if clustering_algo == 'stoch':
            centroids, labels, inertia = StochHierarchical(max_size, weights=importance_scale,
                                                        n_jobs=30).clustering(features, smaller_sizes[l])
        elif clustering_algo == 'stoch_high':
            centroids, labels, inertia = StochHierarchical(max_size, weights=importance_scale,
                                                        n_jobs=30, n_restarts=5000).clustering(features, smaller_sizes[l])
        elif clustering_algo == 'stoch_very_high':
            centroids, labels, inertia = StochHierarchical(max_size, weights=importance_scale,
                                                        n_jobs=30, n_restarts=50000).clustering(features, smaller_sizes[l])
        elif clustering_algo == 'kmeans':
            kmeans = KMeans(n_clusters=smaller_sizes[l], n_init=10 ** 5, max_iter=10 ** 4, tol=10 ** -5).fit(
                features)
            centroids, labels, inertia = kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_
        else:
            centroids, labels, inertia = WeightHierarchical().clustering(features, smaller_sizes[l])

        print('pre inertia', inertia)
        if centroids is not None:
            kmeans = KMeans(n_clusters=n_keep, init=centroids, max_iter=300, tol=1e-4)
            kmeans.fit(features)
            labels = kmeans.labels_
            inertia = kmeans.inertia_
        print('inertia final', inertia)

        # 3. Create Pi Matrix from labels
        # pi[i, j] = 1/N_k if channel i belongs to cluster j, else 0
        pi = np.zeros((n_channels, n_keep))

        for k in range(n_keep):
            # Find all original channels belonging to cluster k
            indices = np.where(labels == k)[0]
            if len(indices) > 0:
                pi[indices, k] = 1.0 / n_channels  # Uniform weight distribution

        pi_list.append(pi)

    return _reconstruct_network(net, layers, pi_list, smaller_sizes, fusion_network=fusion_network)