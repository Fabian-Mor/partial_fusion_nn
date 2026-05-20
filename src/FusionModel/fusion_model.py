import copy
import torch.nn as nn

from src.base_model import BaseModel
import torch
import numpy as np


class FusionModel(BaseModel):
    def __init__(self, model1, model2, method, data=None, lambdas=None, bn_data=None,
                 running_stats_init='recalibrate', folded_bn=False):
        super(FusionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if lambdas is None:
            lambdas = [0.5, 0.5]
        self.models = [model1, model2]
        self.method = method
        self.data = data
        self.lambdas = lambdas
        self.folded_bn = folded_bn
        self.layers1 = model1.get_layer_names()
        self.layers2 = model2.get_layer_names()
        self.layers_with_weights1 = model1.get_layer_names_with_weights()
        self.layers_with_weights2 = model2.get_layer_names_with_weights()
        self.fused_layers = []
        self.input_size = self.models[0].input_size
        self.non_zero_weights = 0
        self.n1 = 0
        self.n2 = 0
        self.n3 = 0
        self.res_layers_merged = {}

        # NEW: detect ResNet (has residual structure)
        res_info = model1.get_residual_layers()
        self.residual_conv_layers = res_info[0]
        self.residual_map = res_info[2] if len(res_info) > 2 else {}
        self.norm_map = res_info[3] if len(res_info) > 3 else {}
        self.identity_residual_map = res_info[4] if len(res_info) > 4 else {}
        self.is_resnet = len(self.residual_conv_layers) > 0 or len(self.identity_residual_map) > 0

        # for ResNets, store fused layers by name for structured forward
        self.fused_layers_dict = {}

        self.fuse_two_models(model1, model2)

        # after main fusion, build identity skip connections and handle BN
        if self.is_resnet:
            self._build_identity_skip_connections(model1)
            if folded_bn:
                # BN is already folded into conv weights — fuse biases, skip BN
                self._fuse_biases(model1, model2)
                self.fused_layers_module = nn.ModuleDict(self.fused_layers_dict)
            else:
                init_rs = running_stats_init if running_stats_init in ('copy', 'cosine') else False
                self._rebuild_batchnorm_layers(model1, model2, init_running_stats=init_rs)
                # correct_isolated_bn_vars attempted to propagate input
                # variance changes to isolated neurons, but the ratio-based
                # correction overcorrects.
                self.fused_layers_module = nn.ModuleDict(self.fused_layers_dict)
                # Only recalibrate if not using analytical running stats
                if running_stats_init == 'recalibrate':
                    recalib_data = bn_data if bn_data is not None else data
                    if recalib_data is not None:
                        self._recalibrate_batchnorm(recalib_data)


    def fuse_two_models(self, model1, model2):
        # Collect layers to skip for ResNets (only BatchNorm — handled separately)
        skip_layers = set()
        if self.is_resnet:
            for name, m in model1.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    skip_layers.add(name)

        for layer1, layer2 in zip(self.layers1, self.layers2):
            if layer1 not in self.layers_with_weights1:
                if not self.is_resnet:
                    self.fused_layers.append(model1.get_layer_by_name(layer1))
                continue
            elif layer1 in skip_layers:
                continue

            A_1, A_2, A_3 = self.method.combine_layers(model1, model2, self.data, self.lambdas, layer1)
            if isinstance(model2.get_layer_by_name(layer2), nn.Linear):
                self.non_zero_weights += A_1.size + A_2.size + A_3.size
                if self.is_resnet:
                    self._fuse_layer_to_dict(A_1, A_2, A_3, layer1, model1, is_linear=True)
                else:
                    self.fuse_linear_layer(A_1, A_2, A_3, layer1)
            elif isinstance(model2.get_layer_by_name(layer2), nn.Conv2d):
                self.non_zero_weights += A_1.size + A_2.size + A_3.size
                if self.is_resnet:
                    self._fuse_layer_to_dict(A_1, A_2, A_3, layer1, model1, is_linear=False)
                else:
                    self.fuse_conv_layer(A_1, A_2, A_3, layer1, model1)
            else:
                print("unknown layer: ", model2.get_layer_by_name(layer2))

        if not self.is_resnet:
            self.fused_layers = torch.nn.ModuleList(self.fused_layers)


    def fuse_conv_layer(self, A_1, A_2, A_3, layer1, model1):
        c_out1, c_in1, kH, kW = A_1.shape
        c_out2, c_in2, _, _ = A_2.shape
        c_out3, c_in3, _, _ = A_3.shape
        m, n = c_out1, c_in1 + c_in2 + c_in3
        m2 = c_out1 - c_out3
        m3 = c_out1 - c_out2
        m1 = abs(c_out1 - c_out3 - c_out2)
        self.n1 = c_in1
        self.n2 = c_in2
        self.n3 = c_in3
        if layer1 != self.layers_with_weights1[-1] and layer1 != self.layers_with_weights1[0]:
            A = np.zeros((m, n, kH, kW), dtype=A_1.dtype)
            A[:, :c_in1] = A_1
            A[:(m1 + m2), c_in1:(c_in1 + c_in2)] = A_2
            A[:m1, (c_in1 + c_in2):] = A_3[:m1]
            A[(m1 + m2):, (c_in1 + c_in2):] = A_3[m1:]
        elif layer1 == self.layers_with_weights1[0]:
            A = A_1
        else:
            m = m1
            A = np.zeros((m, c_in1 + c_in2 + c_in3, kH, kW), dtype=A_1.dtype)
            A[:, :c_in1] = self.final_weighting[0] * A_1
            A[:, c_in1:(c_in1 + c_in2)] = self.final_weighting[1] * A_2
            A[:, (c_in1 + c_in2):] = self.final_weighting[2] * A_3
        pad = model1.get_layer_by_name(layer1).padding
        stride = model1.get_layer_by_name(layer1).stride
        new_layer = nn.Conv2d(in_channels=A.shape[1], out_channels=A.shape[0], kernel_size=(kH, kW), stride=stride,
                              padding=pad, bias=False, device=self.device)
        with torch.no_grad():
            new_layer.weight.copy_(torch.from_numpy(A))
        self.fused_layers.append(new_layer)

    def fill_vector(self, w1, w2, w3, n1, n2, n3):
        n = n1 + n2 + n3
        W = np.zeros(n)
        W[:n1] = w1
        W[n1:n1 + n2] = w2
        W[n1 + n2:n] = w3
        return torch.from_numpy(W)

    def fuse_linear_layer(self, A_1, A_2, A_3, layer1):
        n1, n2, n3 = A_1.shape[1], A_2.shape[1], A_3.shape[1]
        m, m2, m3 = A_1.shape[0], A_1.shape[0] - A_3.shape[0], A_1.shape[0] - A_2.shape[0]
        m1 = abs((A_1.shape[0] - A_3.shape[0] - A_2.shape[0]))
        n = n1 + n2 + n3
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        if layer1 != self.layers_with_weights1[-1] and layer1 != self.layers_with_weights1[0]:
            A = np.zeros((m, n))
            A[:, :n1] = A_1
            A[:(m1 + m2), n1:(n1 + n2)] = A_2
            A[:m1, (n1 + n2):] = A_3[:m1, :]
            A[(m1 + m2):, (n1 + n2):] = A_3[m1:, :]
            weights = torch.from_numpy(A)
            new_layer = nn.Linear(n, m, bias=False, device=self.device)
            with torch.no_grad():
                new_layer.weight.copy_(weights)
            self.fused_layers.append(new_layer)
        elif layer1 == self.layers_with_weights1[0]:
            A = A_1
            weights = torch.from_numpy(A)
            new_layer = nn.Linear(A.shape[1], A.shape[0], bias=False, device=self.device)
            with torch.no_grad():
                new_layer.weight.copy_(weights)
            self.fused_layers.append(new_layer)
        else:
            m = m1
            A = np.zeros((m, n))
            A[:, :n1] = A_1
            A[:, n1:(n1 + n2)] = A_2
            A[:, (n1 + n2):] = A_3
            weights = torch.from_numpy(A)
            new_layer = nn.Linear(n, m, bias=False, device=self.device)
            with torch.no_grad():
                new_layer.weight.copy_(weights)
            self.fused_layers.append(new_layer)


    def _fuse_layer_to_dict(self, A_1, A_2, A_3, layer1, model1, is_linear=False):
        if is_linear:
            if A_2.size == 0 and A_3.size == 0:
                A = A_1
            else:
                n1, n2, n3 = A_1.shape[1], A_2.shape[1], A_3.shape[1]
                n = n1 + n2 + n3
                m = A_1.shape[0]
                A = np.zeros((m, n))
                A[:, :n1] = A_1
                A[:, n1:(n1 + n2)] = A_2
                A[:, (n1 + n2):] = A_3
            new_layer = nn.Linear(A.shape[1], A.shape[0], bias=False, device=self.device)
            with torch.no_grad():
                new_layer.weight.copy_(torch.from_numpy(A))
            self.fused_layers_dict[layer1] = new_layer
            return

        c_out1, c_in1, kH, kW = A_1.shape
        c_out2, c_in2, _, _ = A_2.shape
        c_out3, c_in3, _, _ = A_3.shape
        m, n = c_out1, c_in1 + c_in2 + c_in3
        m2 = c_out1 - c_out3
        m1 = abs(c_out1 - c_out3 - c_out2)

        non_res_non_norm = [l for l in self.layers_with_weights1
                            if l not in set(self.residual_conv_layers)
                            and not isinstance(model1.get_layer_by_name(l),
                                               (nn.BatchNorm2d, nn.BatchNorm1d))]
        is_first = (layer1 == non_res_non_norm[0])

        if is_first:
            A = A_1
        elif A_2.size == 0 and A_3.size == 0:
            # Structured pruning: everything is in A1, no block assembly needed
            A = A_1
        else:
            A = np.zeros((m, n, kH, kW), dtype=A_1.dtype)
            A[:, :c_in1] = A_1
            A[:(m1 + m2), c_in1:(c_in1 + c_in2)] = A_2
            A[:m1, (c_in1 + c_in2):] = A_3[:m1]
            A[(m1 + m2):, (c_in1 + c_in2):] = A_3[m1:]

        pad = model1.get_layer_by_name(layer1).padding
        stride = model1.get_layer_by_name(layer1).stride
        new_layer = nn.Conv2d(in_channels=A.shape[1], out_channels=A.shape[0],
                              kernel_size=(kH, kW), stride=stride, padding=pad,
                              bias=False, device=self.device)
        with torch.no_grad():
            new_layer.weight.copy_(torch.from_numpy(A))
        self.fused_layers_dict[layer1] = new_layer


    def _build_identity_skip_connections(self, model1):
        """
        For non-downsample residual blocks, partial fusion may change the
        channel count/ordering between block input and output.  We create
        explicit 1x1 conv layers from the OT-aligned weights computed in
        base_fusion so that the residual addition is correct.
        """
        if not hasattr(self.method, 'A1_list') or self.method.A1_list is None:
            return

        for skip_name, (first_conv, last_conv) in self.identity_residual_map.items():
            if skip_name not in self.method.A1_list:
                continue

            A_1 = self.method.A1_list[skip_name]
            A_2 = self.method.A2_list[skip_name]
            A_3 = self.method.A3_list[skip_name]

            # Direct skip composition: A1 is the full weight, A2/A3 are empty
            if A_2.size == 0 and A_3.size == 0:
                A = A_1
            else:
                c_out1, c_in1, kH, kW = A_1.shape
                c_out2, c_in2, _, _ = A_2.shape
                c_out3, c_in3, _, _ = A_3.shape
                m, n = c_out1, c_in1 + c_in2 + c_in3
                m2 = c_out1 - c_out3
                m1 = abs(c_out1 - c_out3 - c_out2)

                A = np.zeros((m, n, kH, kW), dtype=A_1.dtype)
                A[:, :c_in1] = A_1
                A[:(m1 + m2), c_in1:(c_in1 + c_in2)] = A_2
                A[:m1, (c_in1 + c_in2):] = A_3[:m1]
                A[(m1 + m2):, (c_in1 + c_in2):] = A_3[m1:]

            # Determine stride: if spatial dims change (downsample block), stride=2
            # For identity blocks, stride is always 1
            new_layer = nn.Conv2d(in_channels=A.shape[1], out_channels=A.shape[0],
                                  kernel_size=1, stride=1, padding=0,
                                  bias=False, device=self.device)
            with torch.no_grad():
                new_layer.weight.copy_(torch.from_numpy(A))
            self.fused_layers_dict[skip_name] = new_layer

    # =====================================================================
    # Rebuild BatchNorm layers using OT alignment information.
    # For fused neurons: permute model A's BN params via ker_for, then
    #   combine with model B's params weighted by lambda.
    # For isolated neurons: use the originating model's BN params directly.
    # Running statistics are overwritten by recalibration afterwards.
    # =====================================================================

    def _rebuild_batchnorm_layers(self, model1, model2, init_running_stats=False):
        lam = self.lambdas[1]

        # Check if the method has OT alignment info
        has_ot_info = (hasattr(self.method, '_layer_to_kernel_idx')
                       and hasattr(self.method, '_mu_f_list'))

        for conv_name, bn_name in self.norm_map.items():
            if conv_name not in self.fused_layers_dict:
                continue

            fused_conv = self.fused_layers_dict[conv_name]
            n_channels = fused_conv.out_channels

            bn1 = model1.get_layer_by_name(bn_name)
            bn2 = model2.get_layer_by_name(bn_name)

            new_bn = nn.BatchNorm2d(n_channels, device=self.device)

            with torch.no_grad():
                # Check if the method provides pre-computed BN params
                # (e.g. from StructuredPruning which extracts them from the ensemble)
                ens_bn = getattr(self.method, '_ensemble_bn_params', {}).get(bn_name)
                if ens_bn is not None and ens_bn['weight'].shape[0] == n_channels:
                    new_bn.weight.data.copy_(ens_bn['weight'].to(self.device))
                    new_bn.bias.data.copy_(ens_bn['bias'].to(self.device))
                    new_bn.running_mean.copy_(ens_bn['running_mean'].to(self.device))
                    new_bn.running_var.copy_(ens_bn['running_var'].to(self.device))
                elif has_ot_info:
                    self._fill_bn_aligned(new_bn, bn1, bn2, conv_name, lam, n_channels,
                                          init_running_stats=init_running_stats)
                else:
                    # Fallback for NaiveFusion: simple weighted average
                    c_orig = bn1.num_features
                    n_fused = min(c_orig, n_channels)
                    if bn1.weight is not None and bn2.weight is not None:
                        new_bn.weight.data[:n_fused] = (1 - lam) * bn1.weight.data[:n_fused] + lam * bn2.weight.data[:n_fused]
                    if bn1.bias is not None and bn2.bias is not None:
                        new_bn.bias.data[:n_fused] = (1 - lam) * bn1.bias.data[:n_fused] + lam * bn2.bias.data[:n_fused]

                new_bn.num_batches_tracked.zero_()

            self.fused_layers_dict[bn_name] = new_bn

    def _fill_bn_aligned(self, new_bn, bn1, bn2, conv_name, lam, n_channels,
                         init_running_stats=False):
        """Fill BN gamma/beta (and optionally running stats) using the OT alignment masks and kernels."""
        import numpy as np
        eps = self.method.eps

        # Determine the kernel output index for this conv layer
        l2k = self.method._layer_to_kernel_idx
        if conv_name in l2k:
            l_out_idx = l2k[conv_name] + 1  # output space = kernel_idx + 1
        elif conv_name in self.residual_map:
            # ds_conv: output space matches last_conv's output
            _, last_conv = self.residual_map[conv_name]
            l_out_idx = l2k[last_conv] + 1
        else:
            # Unknown layer — leave default init (ones/zeros)
            return

        mu_f = self.method._mu_f_list
        mu_i = self.method._mu_i_list
        nu_f = self.method._nu_f_list
        nu_i = self.method._nu_i_list
        ker_fwd = self.method._kernels_forward

        # Output space masks
        mfelp = np.array(mu_f[l_out_idx] > eps).flatten()
        mielp = np.array(mu_i[l_out_idx] > eps).flatten()
        nfelp = np.array(nu_f[l_out_idx] > eps).flatten()
        nielp = np.array(nu_i[l_out_idx] > eps).flatten()

        # Output weight splits (fraction of mass in fused vs isolated)
        mwfp = mu_f[l_out_idx][mfelp].flatten().astype(float)
        mwfp = mwfp / (mwfp + mu_i[l_out_idx][mfelp].flatten().astype(float))
        mwip = mu_i[l_out_idx][mielp].flatten().astype(float)
        mwip = mwip / (mwip + mu_f[l_out_idx][mielp].flatten().astype(float))
        nwfp = nu_f[l_out_idx][nfelp].flatten().astype(float)
        nwfp = nwfp / (nwfp + nu_i[l_out_idx][nfelp].flatten().astype(float))
        nwip = nu_i[l_out_idx][nielp].flatten().astype(float)
        nwip = nwip / (nwip + nu_f[l_out_idx][nielp].flatten().astype(float))

        ker_for = ker_fwd[l_out_idx]

        # Helper: reconstruct a BN parameter vector in fused-model ordering
        def _assemble_param(param_a, param_b):
            pa = param_a.detach().cpu().numpy()
            pb = param_b.detach().cpu().numpy()

            # Fused portion: permute model A, combine with model B
            pa_fused = pa[mfelp] * mwfp
            pb_fused = pb[nfelp] * nwfp
            fused = (1 - lam) * (ker_for @ pa_fused) + lam * pb_fused

            # Model A isolated
            iso_a = pa[mielp] * mwip

            # Model B isolated
            iso_b = pb[nielp] * nwip

            return np.concatenate([fused, iso_a, iso_b])

        # Gamma
        if bn1.weight is not None and bn2.weight is not None:
            gamma = _assemble_param(bn1.weight.data, bn2.weight.data)
            new_bn.weight.data.copy_(torch.from_numpy(gamma).to(new_bn.weight.device))

        # Beta
        if bn1.bias is not None and bn2.bias is not None:
            beta = _assemble_param(bn1.bias.data, bn2.bias.data)
            new_bn.bias.data.copy_(torch.from_numpy(beta).to(new_bn.bias.device))

        # Running stats initialization
        if init_running_stats == 'copy':
            # Simple OT-aligned combination (same formula as gamma/beta)
            rm = _assemble_param(bn1.running_mean, bn2.running_mean)
            new_bn.running_mean.copy_(torch.from_numpy(rm).float().to(new_bn.running_mean.device))
            rv = _assemble_param(bn1.running_var, bn2.running_var)
            new_bn.running_var.copy_(torch.from_numpy(rv).float().to(new_bn.running_var.device))

        elif init_running_stats == 'cosine':
            # Cosine-based variance prediction:
            # var_F = (1-λ)²·var_A + λ²·var_B + 2λ(1-λ)·ρ·√var_A·√var_B
            # where ρ = cosine similarity of aligned weight vectors.
            # mean_F = (1-λ)·mean_A_permuted + λ·mean_B (standard combination)
            mean_a = bn1.running_mean.cpu().numpy()
            mean_b = bn2.running_mean.cpu().numpy()
            var_a = bn1.running_var.cpu().numpy()
            var_b = bn2.running_var.cpu().numpy()

            # Mean: standard OT-aligned combination
            rm = _assemble_param(bn1.running_mean, bn2.running_mean)
            new_bn.running_mean.copy_(torch.from_numpy(rm).float().to(new_bn.running_mean.device))

            # Variance: cosine-based for fused, direct copy for isolated
            # Get weight vectors for cosine computation
            ker_bwd = self.method._kernels_backward
            l2k_local = self.method._layer_to_kernel_idx
            if conv_name in l2k_local:
                k_idx = l2k_local[conv_name]
            elif conv_name in self.residual_map:
                _, last_conv = self.residual_map[conv_name]
                k_idx = l2k_local[last_conv]
            else:
                k_idx = 0
            l_in = k_idx  # input kernel index

            w_a = self.models[0].get_layer_by_name(conv_name)
            w_b = self.models[1].get_layer_by_name(conv_name)

            a_fused_idx = np.where(mfelp)[0]
            b_fused_idx = np.where(nfelp)[0]
            a_iso_idx = np.where(mielp)[0]
            b_iso_idx = np.where(nielp)[0]

            # Input space masks for alignment
            # For ds_conv: input space = first_conv's input (block input),
            # not last_conv's input (block internal space)
            if conv_name in self.residual_map:
                first_conv, _ = self.residual_map[conv_name]
                l_in = l2k_local[first_conv]

            mu_f_local = self.method._mu_f_list
            mfel_in = (mu_f_local[l_in] > eps).flatten()
            nfel_in = (self.method._nu_f_list[l_in] > eps).flatten()
            k_back = ker_bwd[l_in]

            # NOTE: previous version used a dimension-check fallback here for
            # ds_conv layers. Now handled properly via l_in = l2k[first_conv].
            # The previous fallback version (simple weighted avg for ds_conv)
            # already achieved ~92-94% accuracy with zero data — see test_cosine_only.py.

            if w_a is not None and w_b is not None and hasattr(w_a, 'weight'):
                wa_np = w_a.weight.detach().cpu().numpy()
                wb_np = w_b.weight.detach().cpu().numpy()

                # Fused neurons: compute per-neuron cosine similarity
                fused_vars = []
                for j in range(min(ker_for.shape[0], len(b_fused_idx))):
                    a_idx = a_fused_idx[np.argmax(ker_for[j])]
                    b_idx = b_fused_idx[j]
                    va = var_a[a_idx]
                    vb = var_b[b_idx]

                    # Compute aligned cosine
                    wa_j = wa_np[a_idx]
                    wb_j = wb_np[b_idx]
                    if wa_j.ndim >= 3:  # conv
                        wa_f = wa_j[mfel_in]
                        wb_f = wb_j[nfel_in]
                        wa_al = np.einsum('ij,jhw->ihw', k_back.T, wa_f)
                    else:  # linear
                        wa_f = wa_j[mfel_in]
                        wb_f = wb_j[nfel_in]
                        wa_al = k_back.T @ wa_f

                    norm_a = np.linalg.norm(wa_al)
                    norm_b = np.linalg.norm(wb_f)
                    if norm_a > 1e-10 and norm_b > 1e-10:
                        rho = np.dot(wa_al.flatten(), wb_f.flatten()) / (norm_a * norm_b)
                    else:
                        rho = 0.0

                    pred_var = ((1-lam)**2 * va + lam**2 * vb
                                + 2*lam*(1-lam) * rho * np.sqrt(max(va, 0)) * np.sqrt(max(vb, 0)))
                    fused_vars.append(max(pred_var, 0.0))

                # Isolated neurons: copy directly
                iso_a_vars = [var_a[idx] for idx in a_iso_idx]
                iso_b_vars = [var_b[idx] for idx in b_iso_idx]

                rv = np.array(fused_vars + iso_a_vars + iso_b_vars, dtype=np.float32)
                # Apply mass fractions (same weighting as gamma/beta)
                # Fused portion already accounts for both models via the formula
                # Isolated portions use the originating model's variance directly
                # (mass fractions are already 1.0 for fully isolated neurons)
                new_bn.running_var.copy_(torch.from_numpy(rv).to(new_bn.running_var.device))
            else:
                # Fallback
                rv = _assemble_param(bn1.running_var, bn2.running_var)
                new_bn.running_var.copy_(torch.from_numpy(rv).float().to(new_bn.running_var.device))

    def _correct_isolated_bn_vars(self, model1, model2):
        """
        Correct isolated neurons' running_var by propagating the input variance
        change from the previous layer. Uses the ratio of cosine-predicted
        running_var to original running_var at the previous layer as a measure
        of how the input distribution has changed.
        """
        eps = self.method.eps
        l2k = self.method._layer_to_kernel_idx
        mu_f = self.method._mu_f_list
        mu_i = self.method._mu_i_list
        nu_i = self.method._nu_i_list

        # Sequential order of convs by kernel index
        all_convs = sorted(l2k.keys(), key=lambda x: l2k[x])

        # Map each conv to the BN feeding its input
        input_bn_map = {}
        for conv_name in all_convs:
            k = l2k[conv_name]
            for prev_conv, prev_k in l2k.items():
                if prev_k == k - 1 and prev_conv in self.norm_map:
                    input_bn_map[conv_name] = self.norm_map[prev_conv]
                    break

        for conv_name in all_convs:
            if conv_name not in self.norm_map:
                continue
            bn_name = self.norm_map[conv_name]
            if bn_name not in self.fused_layers_dict:
                continue

            k_idx = l2k[conv_name]
            l_out = k_idx + 1
            n_fused = int(np.sum(mu_f[l_out] > eps))
            n_iso_a = int(np.sum(mu_i[l_out] > eps))
            n_iso_b = int(np.sum(nu_i[l_out] > eps))

            if n_iso_a + n_iso_b == 0:
                continue

            if conv_name not in input_bn_map:
                continue
            prev_bn_name = input_bn_map[conv_name]
            if prev_bn_name not in self.fused_layers_dict:
                continue

            # Previous layer: cosine-predicted running_var (fused model)
            prev_var_fused = self.fused_layers_dict[prev_bn_name].running_var.detach().cpu().numpy()

            # Previous layer: original running_var
            prev_bn_a = model1.get_layer_by_name(prev_bn_name)
            prev_bn_b = model2.get_layer_by_name(prev_bn_name)
            if prev_bn_a is None:
                continue
            prev_var_a = prev_bn_a.running_var.detach().cpu().numpy()
            prev_var_b = prev_bn_b.running_var.detach().cpu().numpy()

            # Per-input-channel variance ratio: how has the input changed?
            # The fused model's previous BN has [fused, iso_A, iso_B] channels.
            # For each channel, ratio = cosine_predicted_var / original_var
            # tells us how the BN normalization differs → affects output variance.
            prev_conv_name = None
            for pc, pk in l2k.items():
                if pk == k_idx - 1:
                    prev_conv_name = pc
                    break

            if prev_conv_name is None:
                continue

            prev_l_out = l2k[prev_conv_name] + 1
            prev_n_f = int(np.sum(mu_f[prev_l_out] > eps))
            prev_n_ia = int(np.sum(mu_i[prev_l_out] > eps))
            prev_n_ib = int(np.sum(nu_i[prev_l_out] > eps))

            mfelp_prev = (mu_f[prev_l_out] > eps).flatten()
            mielp_prev = (mu_i[prev_l_out] > eps).flatten()
            nielp_prev = (nu_i[prev_l_out] > eps).flatten()

            # Build per-channel ratio: cosine_var / orig_var
            # This ratio < 1 for fused channels (cosine predicts lower var)
            # This ratio ≈ 1 for isolated channels (we copied orig var)
            # When BN divides by running_var, a lower running_var means
            # larger BN output → the next layer sees larger input.
            # So the relevant ratio for the NEXT layer's input variance is
            # orig_var / cosine_var (inverted): how much larger the BN output is.
            input_var_ratio = np.ones(len(prev_var_fused), dtype=np.float32)

            a_fused_idx_prev = np.where(mfelp_prev)[0]
            a_iso_idx_prev = np.where(mielp_prev)[0]
            b_iso_idx_prev = np.where(nielp_prev)[0]

            for i in range(prev_n_f):
                # Fused channel i: cosine_var vs average of matched originals
                cv = prev_var_fused[i]
                # Approximate original var as average of model A and B
                # (the fused channel is a mix of both)
                ov_a = np.mean(prev_var_a[mfelp_prev]) if np.any(mfelp_prev) else prev_var_a.mean()
                ov_b = np.mean(prev_var_b[mfelp_prev[:len(prev_var_b)]] if len(prev_var_b) >= np.sum(mfelp_prev) else prev_var_b)
                ov = 0.5 * ov_a + 0.5 * ov_b
                if ov > 1e-10:
                    input_var_ratio[i] = cv / ov

            for i in range(prev_n_ia):
                idx = prev_n_f + i
                if idx < len(input_var_ratio) and i < len(a_iso_idx_prev):
                    ov = prev_var_a[a_iso_idx_prev[i]]
                    cv = prev_var_fused[idx]
                    if ov > 1e-10:
                        input_var_ratio[idx] = cv / ov

            for i in range(prev_n_ib):
                idx = prev_n_f + prev_n_ia + i
                if idx < len(input_var_ratio) and i < len(b_iso_idx_prev):
                    ov = prev_var_b[b_iso_idx_prev[i]]
                    cv = prev_var_fused[idx]
                    if ov > 1e-10:
                        input_var_ratio[idx] = cv / ov

            # Apply correction to isolated neurons
            fused_bn = self.fused_layers_dict[bn_name]
            rv = fused_bn.running_var.detach().cpu().numpy().copy()

            def _apply_correction(model, iso_idx_list, start_offset):
                conv = model.get_layer_by_name(conv_name)
                if conv is None or not hasattr(conv, 'weight'):
                    return
                w_all = conv.weight.detach().cpu().numpy()
                for k_i, orig_idx in enumerate(iso_idx_list):
                    neuron_idx = start_offset + k_i
                    if neuron_idx >= len(rv):
                        break
                    w = w_all[orig_idx]
                    w_sq = np.sum(w.reshape(w.shape[0], -1) ** 2, axis=1)
                    n_in = min(len(w_sq), len(input_var_ratio))
                    if n_in > 0 and np.sum(w_sq[:n_in]) > 1e-10:
                        correction = np.sum(w_sq[:n_in] * input_var_ratio[:n_in]) / np.sum(w_sq[:n_in])
                        rv[neuron_idx] *= correction

            mielp = (mu_i[l_out] > eps).flatten()
            nielp_out = (nu_i[l_out] > eps).flatten()
            _apply_correction(model1, np.where(mielp)[0], n_fused)
            _apply_correction(model2, np.where(nielp_out)[0], n_fused + n_iso_a)

            with torch.no_grad():
                fused_bn.running_var.copy_(torch.from_numpy(rv).float().to(fused_bn.running_var.device))

    def _fuse_biases(self, model1, model2):
        """Fuse conv biases for BN-folded models using OT alignment."""
        lam = self.lambdas[1]
        eps = self.method.eps

        has_ot_info = (hasattr(self.method, '_layer_to_kernel_idx')
                       and hasattr(self.method, '_mu_f_list'))
        if not has_ot_info:
            return

        l2k = self.method._layer_to_kernel_idx
        mu_f = self.method._mu_f_list
        mu_i = self.method._mu_i_list
        nu_f = self.method._nu_f_list
        nu_i = self.method._nu_i_list
        ker_fwd = self.method._kernels_forward

        def _assemble_bias(b_a, b_b, l_out_idx):
            mfelp = np.array(mu_f[l_out_idx] > eps).flatten()
            mielp = np.array(mu_i[l_out_idx] > eps).flatten()
            nfelp = np.array(nu_f[l_out_idx] > eps).flatten()
            nielp = np.array(nu_i[l_out_idx] > eps).flatten()

            mwfp = mu_f[l_out_idx][mfelp].flatten().astype(float)
            mwfp = mwfp / (mwfp + mu_i[l_out_idx][mfelp].flatten().astype(float))
            mwip = mu_i[l_out_idx][mielp].flatten().astype(float)
            mwip = mwip / (mwip + mu_f[l_out_idx][mielp].flatten().astype(float))
            nwfp = nu_f[l_out_idx][nfelp].flatten().astype(float)
            nwfp = nwfp / (nwfp + nu_i[l_out_idx][nfelp].flatten().astype(float))
            nwip = nu_i[l_out_idx][nielp].flatten().astype(float)
            nwip = nwip / (nwip + nu_f[l_out_idx][nielp].flatten().astype(float))

            k_for = ker_fwd[l_out_idx]
            fused_part = (1 - lam) * (k_for @ (b_a[mfelp] * mwfp)) + lam * (b_b[nfelp] * nwfp)
            iso_a = b_a[mielp] * mwip
            iso_b = b_b[nielp] * nwip
            return np.concatenate([fused_part, iso_a, iso_b])

        for conv_name, layer in list(self.fused_layers_dict.items()):
            if not isinstance(layer, (nn.Conv2d, nn.Linear)):
                continue

            conv1 = model1.get_layer_by_name(conv_name)
            conv2 = model2.get_layer_by_name(conv_name)
            if conv1 is None or conv2 is None or conv1.bias is None or conv2.bias is None:
                continue

            b_a = conv1.bias.detach().cpu().numpy()
            b_b = conv2.bias.detach().cpu().numpy()

            if conv_name in l2k:
                l_out_idx = l2k[conv_name] + 1
                fused_bias = _assemble_bias(b_a, b_b, l_out_idx)
            elif conv_name in self.residual_map:
                _, last_conv = self.residual_map[conv_name]
                l_out_idx = l2k[last_conv] + 1
                fused_bias = _assemble_bias(b_a, b_b, l_out_idx)
            else:
                fused_bias = (1 - lam) * b_a + lam * b_b

            n_out = layer.weight.shape[0]
            if len(fused_bias) != n_out:
                continue
            with torch.no_grad():
                layer.bias = nn.Parameter(torch.from_numpy(fused_bias).float().to(self.device))

    def _recalibrate_batchnorm(self, data):
        was_training = self.training
        self.train()
        with torch.no_grad():
            if isinstance(data, torch.utils.data.DataLoader):
                for batch in data:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)
                    self.forward(x)
            elif isinstance(data, torch.Tensor):
                batch_size = 64
                for i in range(0, data.shape[0], batch_size):
                    self.forward(data[i:i+batch_size].to(self.device))
        if not was_training:
            self.eval()

    # =====================================================================
    # Forward pass
    # =====================================================================

    def forward(self, x):
        x = x.to(self.device)
        if not self.is_resnet:
            if self.input_size is not None:
                x = x.view(-1, self.input_size)
            for idx, layer in enumerate(self.fused_layers):
                x = layer(x)
            return x
        return self._forward_resnet(x)

    def _forward_resnet(self, x):
        d = self.fused_layers_dict
        def get(name):
            return d.get(name, None)

        x = d['conv1'](x)
        if not self.folded_bn:
            bn1 = get('bn1')
            if bn1 is not None:
                x = bn1(x)
        x = torch.relu(x)

        for stage in range(1, 5):
            for block in range(2):
                prefix = f'layer{stage}_block{block}'
                x = self._forward_basic_block(x, prefix)

        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = d['fc'](x)
        return x

    def _forward_basic_block(self, x, prefix):
        d = self.fused_layers_dict
        identity = x

        out = d[f'{prefix}_conv1'](x)
        if not self.folded_bn:
            bn1 = d.get(f'{prefix}_bn1')
            if bn1 is not None:
                out = bn1(out)
        out = torch.relu(out)

        out = d[f'{prefix}_conv2'](out)
        if not self.folded_bn:
            bn2 = d.get(f'{prefix}_bn2')
            if bn2 is not None:
                out = bn2(out)

        # Explicit downsample skip connection
        ds_conv_name = f'{prefix}_ds_conv'
        if ds_conv_name in d:
            identity = d[ds_conv_name](x)
            if not self.folded_bn:
                ds_bn_name = f'{prefix}_ds_bn'
                if ds_bn_name in d:
                    identity = d[ds_bn_name](identity)
        else:
            # Identity skip connection (created by partial fusion for
            # non-downsample blocks when channel dims expand)
            skip_name = f'{prefix}_skip'
            if skip_name in d:
                identity = d[skip_name](x)

        out = out + identity
        out = torch.relu(out)
        return out

    # =====================================================================
    # Utility methods
    # =====================================================================

    def get_layer_names(self):
        layer_names = []
        for name, layer in self.named_modules():
            if name != '' and name != 'fused_layers' and name != 'fused_layers_module':
                layer_names.append(name)
        return layer_names

    def get_layer_names_with_weights(self):
        layer_names = []
        for name, layer in self.named_modules():
            if name != '' and name != 'fused_layers' and name != 'fused_layers_module' and hasattr(layer, 'weight'):
                layer_names.append(name)
        return layer_names

    def get_prev_fused_weights(self):
        if self.is_resnet:
            last_conv = None
            for name, mod in self.fused_layers_dict.items():
                if isinstance(mod, nn.Conv2d):
                    last_conv = name
            if last_conv:
                return self.fused_layers_dict[last_conv].weight.detach().cpu().numpy()
        if len(self.fused_layers) > 0:
            counter = -1
            while not hasattr(self.fused_layers[counter], 'weight'):
                counter -= 1
            return self.fused_layers[counter].weight.detach().cpu().numpy()
        else:
            raise Exception("No layers were fused so far")
