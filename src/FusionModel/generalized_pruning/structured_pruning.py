from src.FusionModel.fusion_model import FusionModel
from src.FusionModel.fusion_methods import PartialFusion
from src.CNN.VGG import _reconstruct_network, VGG11
import torch.nn as nn
import numpy as np
import torch
import ot


class StructuredPruning:
    def __init__(self, alphas=None):
        self.A1_list = None
        self.A2_list = None
        self.A3_list = None
        if alphas is None:
            self.alphas = [0]
        elif not isinstance(alphas, list):
            self.alphas = [alphas]
        else:
            self.alphas = alphas

    def combine_layers(self, model1, model2, data, lambdas, layer):
        if self.A1_list is None or self.A2_list is None or self.A3_list is None:
            A1, A2, A3 = self._prune(model1, model2, data=data, lambdas=lambdas)
            self.A1_list = A1
            self.A2_list = A2
            self.A3_list = A3
        return self.A1_list[layer], self.A2_list[layer], self.A3_list[layer]

    def prune_network(self, model, model_size, mult, model_cls, data, post_proc=False, which_act=0):
        if model_cls is VGG11:
            return self._prune_cnn(model, model_size, mult, post_proc=post_proc)
        else:
            return self._prune_ff(model, model_size, mult, model_cls, data, post_proc=post_proc, which_act=which_act)

    def _prune_ff(self, model, model_size, mult, model_cls, data, which_act=0, post_proc=False):
        smaller_sizes = [int(np.round(mult * l)) for l in model_size]
        layers1 = model.get_layer_names_with_weights()
        L = len(layers1)
        pre_clusters = []
        for l in range(L):
            if data is not None:
                w_a = model.get_activations(layers1[l], data, numpy=True).T
            else:
                if l < L - 1:
                    w_a = model.get_incoming_weights(layers1[l + 1], numpy=True).T
            n_w, m_w = model.get_incoming_weights(layers1[l], numpy=True).shape
            if l == 0:
                whinit = model.get_incoming_weights(layers1[l], numpy=True)
                n_w, m_w = whinit.shape
                inds_in = np.arange(0, m_w)
                pre_clusters.append(inds_in.copy())
            else:
                inds_in = inds_out.copy()
            if l == L - 1:
                inds_out = np.arange(0, n_w)
            else:
                if data is not None:
                    norm_h = np.std(w_a, axis=1)
                else:
                    norm_h = np.linalg.norm(w_a, axis=1)
                sort_inds = np.argsort(norm_h)[::-1]
                inds_out = sort_inds[:smaller_sizes[l]]
            pre_clusters.append(inds_out.copy())

        if post_proc:
            input_length = model.get_incoming_weights(layers1[0], numpy=True).shape[1]
            kernels_forward = [np.identity(input_length)]
            kernels_backward = [np.identity(input_length)]
            if which_act == 0:
                act_h = torch.nn.ReLU()
            elif which_act == 1:
                act_h = torch.nn.LeakyReLU()
            elif which_act == 2:
                act_h = torch.nn.GELU()
            for l in range(L - 1):
                w_out = model.get_incoming_weights(layers1[l + 1], numpy=True).copy()
                layer_name = layers1[l]
                activations = model.get_activations(layer_name, data).T
                activations = act_h(activations)
                activations = activations.detach().cpu().numpy()
                state = activations
                x_full = state
                x_centers = state[pre_clusters[l + 1], :]
                c = ot.dist(x_full, x_centers, metric='sqeuclidean')
                w_full = np.ones(len(state)) / len(state)
                w_centers = np.ones(len(x_centers)) / len(x_centers)
                pi_t = ot.emd(w_full, w_centers, c)
                w_mu = np.sum(pi_t, axis=1)
                w_quant = np.sum(pi_t, axis=0)
                k_for = np.transpose(pi_t / w_mu[:, None])
                k_back = pi_t / w_quant[None, :]
                kernels_forward.append(k_for.copy())
                kernels_backward.append(k_back.copy())
            kernels_forward.append(np.identity(len(w_out)))
            kernels_backward.append(np.identity(len(w_out)))
            weight_list = []
            for l in range(L):
                k_back_incoming = kernels_backward[l]
                k_for_outgoing = kernels_forward[l + 1]
                w_a = model.get_incoming_weights(layers1[l], numpy=True)
                w_b = k_for_outgoing @ w_a @ k_back_incoming
                weight_list.append(w_b.copy())
            out_net = model_cls(hidden_size_1=smaller_sizes[0], hidden_size_2=smaller_sizes[1],
                                hidden_size_3=smaller_sizes[2], assign_w=True, w_mats=weight_list, which_act=which_act)
        else:
            weight_list = []
            for l in range(L):
                w_a = model.get_incoming_weights(layers1[l], numpy=True)
                w_b = w_a[:, pre_clusters[l]][pre_clusters[l + 1], :]
                weight_list.append(w_b.copy())
            out_net = model_cls(hidden_size_1=smaller_sizes[0], hidden_size_2=smaller_sizes[1],
                                hidden_size_3=smaller_sizes[2], assign_w=True, w_mats=weight_list, which_act=which_act)
        return out_net

    def _prune_cnn(self, model, model_size, mult, only_conv=True, post_proc=False):
        smaller_sizes = [int(np.round(mult * l)) for l in model_size]
        all_layers = model.get_layer_names_with_weights()
        target_layers = [l for l in all_layers if
                         len(model.get_incoming_weights(l, numpy=True).shape) == 4] if only_conv else all_layers
        pi_list = []
        for l, layer_name in enumerate(target_layers):
            w_curr = model.get_incoming_weights(layer_name, numpy=True)
            n_channels = w_curr.shape[0]
            if l >= len(smaller_sizes):
                pi = np.eye(n_channels)
                pi_list.append(pi)
                continue
            n_keep = smaller_sizes[l]
            importance = np.linalg.norm(w_curr.reshape(n_channels, -1), axis=1)
            top_indices = np.argsort(importance)[::-1][:n_keep]
            if post_proc:
                X_full = w_curr.reshape(n_channels, -1)
                norms = np.linalg.norm(X_full, axis=1, keepdims=True)
                X_full = np.divide(X_full, norms + 1e-9, out=np.zeros_like(X_full), where=norms != 0)
                X_centers = X_full[top_indices]
                M = ot.dist(X_full, X_centers, metric='sqeuclidean')
                M = M / (M.max() + 1e-9)
                w_source = np.ones(n_channels) / n_channels
                w_target = np.ones(n_keep) / n_keep
                pi = ot.emd(w_source, w_target, M)
                pi_list.append(pi)
            else:
                pi = np.zeros((n_channels, n_keep))
                for new_idx, old_idx in enumerate(top_indices):
                    pi[old_idx, new_idx] = 1.0
                pi_list.append(pi)
        return _reconstruct_network(model, target_layers, pi_list, smaller_sizes)

    # =====================================================================
    # Main pruning method: prune an ensemble of two models
    # =====================================================================

    def _prune(self, model1, model2, data, lambdas):
        """
        Build an ensemble of model1 and model2 (alpha=1), then prune it
        to a target size determined by self.alphas.

        Works for both VGG/MLP and ResNet architectures.
        """
        # Detect if this is a ResNet
        res_info = model1.get_residual_layers()
        is_resnet = len(res_info[0]) > 0 or (len(res_info) > 4 and len(res_info[4]) > 0)

        if is_resnet:
            return self._prune_resnet(model1, model2, data, lambdas)
        else:
            return self._prune_standard(model1, model2, data, lambdas)

    def _prune_standard(self, model1, model2, data, lambdas):
        """Original pruning logic for VGG/MLP."""
        net = FusionModel(model1, model2, PartialFusion(alphas=1), lambdas=lambdas)
        pre_clusters = self.compute_pre_cluster(net, data, lambdas)
        net_layer = net.get_layer_names_with_weights()
        A1 = {}
        A2 = {}
        A3 = {}
        layers = model1.get_layer_names_with_weights()
        L = len(layers)
        layer1 = layers[0]
        w_pre = model1.get_incoming_weights(layer1, numpy=True)
        idx_pre2 = [x for x in range(w_pre.shape[1])]
        idx_pre3 = [x for x in range(w_pre.shape[1])]

        for i, layer in enumerate(layers):
            w = net.get_incoming_weights(net_layer[i], numpy=True)
            w1 = model1.get_incoming_weights(layer, numpy=True)
            w2 = model2.get_incoming_weights(layer, numpy=True)
            cluster = pre_clusters[i + 1]
            idx2 = [x for x in cluster if x < w1.shape[0]]
            idx3 = [x - w1.shape[0] for x in cluster if x >= w1.shape[0]]
            idx1 = len(idx2) + len(idx3)
            if len(w.shape) == 2:
                if i == L - 1:
                    A1[layer] = np.zeros((idx1, 0))
                    A2[layer] = lambdas[0] * w1[:, idx_pre2]
                    A3[layer] = lambdas[1] * w2[:, idx_pre3]
                elif i == 0:
                    A1[layer] = np.concatenate([w1[idx2, :][:, idx_pre2], w2[idx3, :][:, idx_pre3]], axis=0)
                    A2[layer] = w1[idx2, :][:, idx_pre2]
                    A3[layer] = w2[idx3, :][:, idx_pre3]
                else:
                    A1[layer] = np.zeros((idx1, 0))
                    A2[layer] = w1[idx2, :][:, idx_pre2]
                    A3[layer] = w2[idx3, :][:, idx_pre3]
            else:
                A1[layer] = np.zeros((len(pre_clusters[i]), 0, w.shape[2], w.shape[3]))
                A2[layer] = w1[idx2, :, :, :]
                A3[layer] = w2[idx3, :, :, :]
            idx_pre2 = idx2
            idx_pre3 = idx3
        return A1, A2, A3

    def _prune_resnet(self, model1, model2, data, lambdas):
        """
        Pruning for ResNet: build ensemble, rank neurons by importance, keep top fraction.

        Unlike partial OT fusion, structured pruning works directly on the ensemble
        weights: we select rows (output neurons) and columns (input neurons) from
        the ensemble's weight matrices. The result goes entirely into A1 (the full
        fused weight), with A2 and A3 empty. This avoids the block-diagonal zero
        structure that would arise from splitting into per-model parts.
        """
        res_info = model1.get_residual_layers()
        residual_conv_layers = set(res_info[0])
        residual_map = res_info[2] if len(res_info) > 2 else {}
        identity_residual_map = res_info[4] if len(res_info) > 4 else {}
        norm_layers = set()
        for name, m in model1.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                norm_layers.add(name)

        # Main conv/linear layers (excluding BN and residual convs)
        main_layers = [l for l in model1.get_layer_names_with_weights()
                       if l not in residual_conv_layers and l not in norm_layers]
        L = len(main_layers)

        # Build ensemble
        net = FusionModel(model1, model2, PartialFusion(alphas=1),
                          lambdas=lambdas,
                          bn_data=data if isinstance(data, torch.utils.data.DataLoader) else None)

        # Compute which neurons to keep per layer (in ensemble indexing)
        pre_clusters = self._compute_resnet_clusters(net, main_layers, data, lambdas)

        A1 = {}
        A2 = {}
        A3 = {}

        for i, layer in enumerate(main_layers):
            # Get the ensemble's fused weight for this layer
            if layer not in net.fused_layers_dict:
                continue
            w_ens = net.fused_layers_dict[layer].weight.detach().cpu().numpy()

            cluster_in = pre_clusters[i]      # input neuron indices to keep
            cluster_out = pre_clusters[i + 1]  # output neuron indices to keep

            is_first = (i == 0)
            is_last = (i == L - 1)

            if len(w_ens.shape) == 4:
                kH, kW = w_ens.shape[2], w_ens.shape[3]
                if is_first:
                    # First layer: no input pruning, select output channels
                    A1[layer] = w_ens[cluster_out, :, :, :]
                else:
                    A1[layer] = w_ens[np.ix_(cluster_out, cluster_in)]

                A2[layer] = np.zeros((0, 0, kH, kW), dtype=w_ens.dtype)
                A3[layer] = np.zeros((0, 0, kH, kW), dtype=w_ens.dtype)

            elif len(w_ens.shape) == 2:
                if is_last:
                    # Last layer: keep all output neurons, select input
                    A1[layer] = w_ens[:, cluster_in]
                else:
                    A1[layer] = w_ens[np.ix_(cluster_out, cluster_in)]

                A2[layer] = np.zeros((0, 0), dtype=w_ens.dtype)
                A3[layer] = np.zeros((0, 0), dtype=w_ens.dtype)

        # Handle downsample conv layers
        for ds_layer, (first_conv, last_conv) in residual_map.items():
            if ds_layer not in net.fused_layers_dict:
                continue
            w_ens = net.fused_layers_dict[ds_layer].weight.detach().cpu().numpy()
            kH, kW = w_ens.shape[2], w_ens.shape[3]

            first_idx = main_layers.index(first_conv)
            last_idx = main_layers.index(last_conv)
            cluster_in = pre_clusters[first_idx]       # input to the block
            cluster_out = pre_clusters[last_idx + 1]    # output of the block

            A1[ds_layer] = w_ens[np.ix_(cluster_out, cluster_in)]
            A2[ds_layer] = np.zeros((0, 0, kH, kW), dtype=w_ens.dtype)
            A3[ds_layer] = np.zeros((0, 0, kH, kW), dtype=w_ens.dtype)

        # Handle identity skip connections
        for skip_name, (first_conv, last_conv) in identity_residual_map.items():
            first_idx = main_layers.index(first_conv)
            last_idx = main_layers.index(last_conv)
            cluster_in = pre_clusters[first_idx]
            cluster_out = pre_clusters[last_idx + 1]

            if skip_name in net.fused_layers_dict:
                # Ensemble has an explicit skip conv — prune it
                w_ens = net.fused_layers_dict[skip_name].weight.detach().cpu().numpy()
                A1[skip_name] = w_ens[np.ix_(cluster_out, cluster_in)]
            else:
                # No explicit skip in ensemble (identity) — build from identity
                n_in = len(cluster_in)
                n_out = len(cluster_out)
                W = np.zeros((n_out, n_in, 1, 1), dtype=np.float32)
                in_list = sorted(cluster_in)
                out_list = sorted(cluster_out)
                in_set = {v: i for i, v in enumerate(in_list)}
                for j, out_idx in enumerate(out_list):
                    if out_idx in in_set:
                        W[j, in_set[out_idx], 0, 0] = 1.0
                A1[skip_name] = W

            A2[skip_name] = np.zeros((0, 0, 1, 1), dtype=np.float32)
            A3[skip_name] = np.zeros((0, 0, 1, 1), dtype=np.float32)

        # Extract BN params from the ensemble for each pruned layer.
        # The ensemble's BN has correct params for the ensemble channel ordering.
        # For the pruned model, we subset these params by the kept neuron indices.
        self._ensemble_bn_params = {}
        norm_map = res_info[3] if len(res_info) > 3 else {}
        all_residual_maps = dict(residual_map)
        all_residual_maps.update(identity_residual_map)

        for conv_name, bn_name in norm_map.items():
            # Determine which cluster indices apply to this BN's channels
            if conv_name in main_layers:
                idx = main_layers.index(conv_name)
                cluster = pre_clusters[idx + 1]
            elif conv_name in all_residual_maps:
                _, last_conv = all_residual_maps[conv_name]
                last_idx = main_layers.index(last_conv)
                cluster = pre_clusters[last_idx + 1]
            else:
                continue

            # Get ensemble BN params
            ens_bn = net.fused_layers_dict.get(bn_name)
            if ens_bn is None or not isinstance(ens_bn, nn.BatchNorm2d):
                continue

            # Subset to kept channels (clusters are already sorted)
            self._ensemble_bn_params[bn_name] = {
                'weight': ens_bn.weight.data[cluster].cpu().clone(),
                'bias': ens_bn.bias.data[cluster].cpu().clone(),
                'running_mean': ens_bn.running_mean[cluster].cpu().clone(),
                'running_var': ens_bn.running_var[cluster].cpu().clone(),
            }

        return A1, A2, A3

    def _compute_resnet_clusters(self, net, main_layers, data, lambdas):
        """
        Compute which neurons to keep in each layer of the ensemble network.

        For each hidden layer, rank neurons by importance (L2 norm of outgoing
        weights or std of activations) and keep the top fraction determined
        by alpha. The first and last layers keep all neurons.

        Returns a list of index arrays, one per layer boundary:
            pre_clusters[0] = input indices (all)
            pre_clusters[i+1] = output indices to keep for layer i
        """
        pre_clusters = []

        # Get the ensemble's layer names with weights (from fused_layers_dict)
        ens_conv_names = [name for name in net.fused_layers_dict
                          if isinstance(net.fused_layers_dict[name], (nn.Conv2d, nn.Linear))]

        # Build ordered list matching main_layers
        # (main_layers are from model1, ens_conv_names are from the fused model)
        # They should be in the same order since fuse_two_models iterates layers1

        for l, layer in enumerate(main_layers):
            # Get the ensemble weight for this layer
            if layer in net.fused_layers_dict:
                w_ens = net.fused_layers_dict[layer].weight.detach().cpu().numpy()
            else:
                # Fallback: use original model weights
                w_ens = net.get_incoming_weights(layer, numpy=True) if hasattr(net, 'get_incoming_weights') else None
                if w_ens is None:
                    continue

            n_neurons = w_ens.shape[0]

            if l == 0:
                # First layer: keep all input channels
                n_in = w_ens.shape[1]
                pre_clusters.append(np.arange(n_in))

            if l == len(main_layers) - 1:
                # Last layer: keep all output neurons
                pre_clusters.append(np.arange(n_neurons))
            else:
                # Compute importance: L2 norm of each neuron's weights
                if len(w_ens.shape) == 4:
                    importance = np.linalg.norm(w_ens.reshape(n_neurons, -1), axis=1)
                else:
                    importance = np.linalg.norm(w_ens, axis=1)

                # Apply lambda weighting: neurons from model A (first half) get
                # weighted by lambdas[0], from B (second half) by lambdas[1]
                if lambdas is not None:
                    # In the ensemble, model A's neurons come first, model B's second
                    # Each model originally has n_neurons/2 neurons
                    split = n_neurons // 2
                    weights = np.ones(n_neurons)
                    weights[:split] = lambdas[0] + 1e-15
                    weights[split:] = lambdas[1] + 1e-15
                    importance = importance * weights

                # Target size: (n_neurons / 2) * (1 + alpha)
                alpha = self.alphas[l % len(self.alphas)]
                target_size = int((n_neurons / 2) * (1 + alpha))
                target_size = min(target_size, n_neurons)

                top_indices = np.sort(np.argsort(importance)[::-1][:target_size])
                pre_clusters.append(top_indices.copy())

        return pre_clusters

    def compute_pre_cluster(self, net, data, lambdas):
        """Original cluster computation for VGG/MLP."""
        layers1 = net.get_layer_names_with_weights()
        L = len(layers1)
        pre_clusters = []
        for l in range(L):
            if data is not None:
                w_a = net.get_activations(layers1[l], data, numpy=True).T
            else:
                if l < L - 1:
                    w_a = net.get_incoming_weights(layers1[l + 1], numpy=True).T
            n_w, m_w = net.get_incoming_weights(layers1[l], numpy=True).shape
            if l == 0:
                whinit = net.get_incoming_weights(layers1[l], numpy=True)
                n_w, m_w = whinit.shape
                inds_in = np.arange(0, m_w)
                pre_clusters.append(inds_in.copy())
            else:
                inds_in = inds_out.copy()
            if l == L - 1:
                inds_out = np.arange(0, n_w)
            else:
                if data is not None:
                    norm_h = np.std(w_a, axis=1)
                else:
                    norm_h = np.linalg.norm(w_a, axis=1)
                alpha = self.alphas[l % len(self.alphas)]
                smaller_size = int((norm_h.shape[0] / 2) * (1 + alpha))
                if lambdas is not None:
                    self.weights = np.ones(norm_h.shape[0])
                    split_point = int(norm_h.shape[0] / 2)
                    self.weights[:split_point] = lambdas[0] + 10 ** -15
                    self.weights[split_point:] = lambdas[1] + 10 ** -15
                    norm_h = norm_h * self.weights
                sort_inds = np.argsort(norm_h)[::-1]
                inds_out = sort_inds[:smaller_size]
            pre_clusters.append(inds_out.copy())
        return pre_clusters
