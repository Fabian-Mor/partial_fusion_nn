import sys
import numpy as np

sys.path.append("../")
from src.base_model import BaseModel


class BaseFusion:
    def __init__(self, eps=10 ** -8, act=False, fix_mu=False, combine_costs=False, pgd=False,
                 tied_permutations=False, direct_skip_composition=False,
                 soft_tie_downsample=False):
        self.A1_list = None
        self.A2_list = None
        self.A3_list = None
        self.eps = eps
        self.act = act
        self.kernel_forward = []
        self.kernel_backward = []
        self.mfels = []
        self.miels = []
        self.nfels = []
        self.niels = []
        self.lambdas_l = []
        self.fix_mu = fix_mu
        self.combine_costs = combine_costs
        self.pgd = pgd
        self.tied_permutations = tied_permutations
        self.direct_skip_composition = direct_skip_composition
        self.soft_tie_downsample = soft_tie_downsample
        self.decoupled = False  # set via PartialFusion
        self.hierarchical = False  # set via PartialFusion
        self._force_isolated = {}

    def _reset(self):
        self.A1_list = None
        self.A2_list = None
        self.A3_list = None

    def combine_layers(self, model1, model2, data, lambdas, layer):
        if self.A1_list is None or self.A2_list is None or self.A3_list is None:
            A1, A2, A3 = self._fuse_two_models_partial([model1, model2], data=data, lambdas=lambdas, pgd=self.pgd)
            self.A1_list = A1
            self.A2_list = A2
            self.A3_list = A3
        return self.A1_list[layer], self.A2_list[layer], self.A3_list[layer]

    def _initialize_fusion(self, models: [BaseModel], lambdas=None, pgd=False):
        K = len(models)
        layers1 = [layer for layer in models[0].get_layer_names_with_weights()
                   if layer not in models[0].get_residual_layers()[0]]
        layers2 = [layer for layer in models[1].get_layer_names_with_weights()
                   if layer not in models[1].get_residual_layers()[0]]
        L = len(layers1)

        init_dimension = models[0].get_incoming_weights(layers1[0], numpy=True).shape[1]
        if lambdas is None:
            inner_list = [1.0 / K] * K
            lambdas = [inner_list for _ in range(L)]
            lambdas_lenght = L
        elif not isinstance(lambdas[0], list):
            lambdas = [lambdas]
            lambdas_lenght = 1
        else:
            lambdas_lenght = len(lambdas)

        #zipped_lists = list(zip(lambdas[-1], models))
        #zipped_lists.sort(key=lambda x: x[0])
        #sorted_lambdas, sorted_models = zip(*zipped_lists)
        #lambdas[-1] = list(sorted_lambdas)
        #models = list(sorted_models)

        if not pgd:
            # initial values are for the input layer, by definition co-monotone coupling
            mu_f_list = [np.ones(init_dimension) / init_dimension]
            nu_f_list = [np.ones(init_dimension) / init_dimension]
            mu_i_list = [np.zeros(init_dimension)]
            nu_i_list = [np.zeros(init_dimension)]
            kernels_forward = [np.identity(init_dimension)]
            kernels_backward = [np.identity(init_dimension)]
        else:
            # initialization of all kernels and all lists
            kernels_forward = []
            kernels_backward = []
            mu_f_list = []
            nu_f_list = []
            mu_i_list = []
            nu_i_list = []
            layers1_no_norm = []
            layers2_no_norm = []
            for l in range(L):
                w = self._get_support(models[0], layers1[l])
                if len(w.shape) == 1:
                    continue
                else:
                    layers1_no_norm.append(layers1[l])
                    layers2_no_norm.append(layers2[l])
                dim = w.shape[1]
                kernels_forward.append(np.identity(dim))
                kernels_backward.append(np.identity(dim))
                mu_f_list.append(np.ones(dim) / dim)
                nu_f_list.append(np.ones(dim) / dim)
                mu_i_list.append(np.zeros(dim) / dim)
                nu_i_list.append(np.zeros(dim) / dim)
            layers1 = layers1_no_norm
            layers2 = layers2_no_norm
            L = len(layers1)
            # append last layer
            w = self._get_support(models[0], layers1[-1])
            dim = w.shape[0]
            kernels_forward.append(np.identity(dim))
            kernels_backward.append(np.identity(dim))
            mu_f_list.append(np.ones(dim) / dim)
            nu_f_list.append(np.ones(dim) / dim)
            mu_i_list.append(np.zeros(dim) / dim)
            nu_i_list.append(np.zeros(dim) / dim)
        return (models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                kernels_forward, kernels_backward, lambdas, lambdas_lenght)

    def _align(self, mu, nu, mu_f, nu_f, ker_adj):
        #CNN alignment
        if len(mu.shape) == 4:
            mu_f = np.array(mu_f).flatten().tolist()
            nu_f = np.array(nu_f).flatten().tolist()
            if len(mu_f) == mu.shape[1]:
                mu = mu[:, mu_f, :, :]
                nu = nu[:, nu_f, :, :]
                mu = np.einsum('bchw,cd->bdhw', mu, ker_adj)  # mu @ k_adjust
                mu = mu.reshape(mu.shape[0], mu.shape[1] * mu.shape[2] * mu.shape[3])
                nu = nu.reshape(nu.shape[0], nu.shape[1] * nu.shape[2] * nu.shape[3])
            else:
                mu = mu.reshape(mu.shape[0], mu.shape[1] * mu.shape[2] * mu.shape[3])
                nu = nu.reshape(nu.shape[0], nu.shape[1] * nu.shape[2] * nu.shape[3])
                mu = mu[:, mu_f]
                nu = nu[:, nu_f]
                mu = mu @ ker_adj
            return mu, nu

        if mu_f.shape[0] != mu.shape[1]:
            # CNN to linear layer
            k = int(mu.shape[1] / mu_f.shape[0])
            mu_f = [x for x in mu_f for _ in range(k)]
            nu_f = [x for x in nu_f for _ in range(k)]
            n = ker_adj.shape[0]
            I = np.eye(k)
            ker_back_extended = ker_adj[:, :, None, None] * I[None, None, :, :]

            # Rearrange to shape (n*k, n*k)
            ker_back_extended = ker_back_extended.transpose(0, 2, 1, 3).reshape(n * k, n * k)
            ker_adj = ker_back_extended

        mu_f = np.array(mu_f).flatten().tolist()
        nu_f = np.array(nu_f).flatten().tolist()
        mu = mu[:, mu_f] @ ker_adj
        nu = nu[:, nu_f]
        return mu, nu

    def _compute_kernels_pcd(self, models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                             kernels_forward, kernels_backward, iter=10):
        # Fix A: build tied kernel pairs from identity skip connections
        tied_to = {}  # maps target kernel index -> source kernel index
        if self.tied_permutations:
            res_info = models[0].get_residual_layers()
            identity_residual_map = res_info[4] if len(res_info) > 4 else {}
            layer_pos = {name: i for i, name in enumerate(layers1)}
            #print(identity_residual_map)
            #print(layer_pos)
            for skip_name, (first_conv, last_conv) in identity_residual_map.items():
                if first_conv in layer_pos and last_conv in layer_pos:
                    l_in = layer_pos[first_conv]      # input kernel index
                    l_out = layer_pos[last_conv] + 1   # output kernel index
                    tied_to[l_out] = l_in
            # Transitive closure: if 3->1 and 5->3, then 5->1
            for target in sorted(tied_to.keys()):
                source = tied_to[target]
                while source in tied_to:
                    source = tied_to[source]
                tied_to[target] = source

        # Build reverse mapping: source -> list of targets tied to it
        tied_from = {}
        for target, src in tied_to.items():
            tied_from.setdefault(src, []).append(target)
        #print(tied_from)

        # Soft tying for downsample blocks: augment the OT cost at l_in and
        # l_out with ds_conv's own supports, but do NOT write into tied_to
        # (kernels at l_in and l_out stay free to differ — ds_conv absorbs
        # any permutation mismatch on the skip path). Each entry maps a
        # boundary position to a list of (ds_layer, other_pos, role) where
        # role is 'in' (this position is l_in) or 'out' (this is l_out).
        soft_tie_info = {}
        if self.soft_tie_downsample:
            res_info = models[0].get_residual_layers()
            ds_residual_map = res_info[2] if len(res_info) > 2 else {}
            layer_pos_local = {name: i for i, name in enumerate(layers1)}
            for ds_layer, (first_conv, last_conv) in ds_residual_map.items():
                if first_conv in layer_pos_local and last_conv in layer_pos_local:
                    l_in_ds = layer_pos_local[first_conv]
                    l_out_ds = layer_pos_local[last_conv] + 1
                    soft_tie_info.setdefault(l_in_ds, []).append((ds_layer, l_out_ds, 'in'))
                    soft_tie_info.setdefault(l_out_ds, []).append((ds_layer, l_in_ds, 'out'))

        for i in range(iter):
            for l in range(L-1):
                mu_for = self._get_support(models[0], layers1[l])
                nu_for = self._get_support(models[1], layers2[l])
                mu_f = mu_f_list[l] > 10 ** -8
                nu_f = nu_f_list[l] > 10 ** -8
                mu_for, nu_for = self._align(mu_for, nu_for, mu_f, nu_f, kernels_backward[l])
                mu_back = self._get_support(models[0], layers1[l + 1], reverse=True)
                nu_back = self._get_support(models[1], layers2[l + 1], reverse=True)
                mu_f = mu_f_list[l+2] > 10 ** -8
                nu_f = nu_f_list[l+2] > 10 ** -8
                mu_back, nu_back = self._align(mu_back, nu_back, mu_f, nu_f, kernels_backward[l + 2])
                if mu_for.shape[0] != mu_back.shape[0]:
                    k = mu_back.shape[0] // mu_for.shape[0]
                    mu_back_new = np.zeros((mu_for.shape[0], mu_back.shape[1]))
                    nu_back_new = np.zeros((nu_for.shape[0], nu_back.shape[1]))
                    for j in range(mu_for.shape[0]):
                        mu_back_new[j] = np.mean(mu_back[j * k:(j + 1) * k, :], axis=0)
                        nu_back_new[j] = np.mean(nu_back[j * k:(j + 1) * k, :], axis=0)
                    mu_back = mu_back_new
                    nu_back = nu_back_new
                mu = [mu_for, mu_back]
                nu = [nu_for, nu_back]
                # Skip-aware cost: if this kernel position (l+1) is the source
                # for tied targets, include forward support from the partner
                # layers so the OT cost accounts for the skip constraint.
                if self.tied_permutations and (l+1) in tied_from:
                    for target_k in tied_from[l+1]:
                        # target_k is a tied output kernel index; the layer
                        # whose input is at target_k-1 provides extra cost info
                        partner_l = target_k - 1
                        if partner_l < L:
                            mu_extra = self._get_support(models[0], layers1[partner_l])
                            nu_extra = self._get_support(models[1], layers2[partner_l])
                            mu_extra_back = self._get_support(models[0], layers1[partner_l + 1], reverse=True)
                            nu_extra_back = self._get_support(models[1], layers2[partner_l + 1], reverse=True)
                            mu_extra, nu_extra = self._align(
                                mu_extra, nu_extra, mu_f_list[partner_l] > 10**-8,
                                nu_f_list[partner_l] > 10**-8, kernels_backward[partner_l])
                            mu_f = mu_f_list[partner_l + 2] > 10 ** -8
                            nu_f = nu_f_list[partner_l + 2] > 10 ** -8
                            mu_extra_back, nu_extra_back = self._align(
                                mu_extra_back, nu_extra_back, mu_f, nu_f,
                                kernels_backward[partner_l + 2])
                            mu.append(mu_extra)
                            mu.append(mu_extra_back)
                            nu.append(nu_extra)
                            nu.append(nu_extra_back)
                # Soft skip-aware cost for downsample blocks: append ds_conv's
                # own support to the OT problem at l_in (using ds_conv's
                # backward support, rows = l_in) and at l_out (forward
                # support, rows = l_out). No tying is enforced — kernels at
                # l_in and l_out are still solved independently.
                if self.soft_tie_downsample and (l+1) in soft_tie_info:
                    for ds_layer, other_pos, role in soft_tie_info[l+1]:
                        if other_pos >= len(kernels_backward):
                            continue
                        reverse = (role == 'in')
                        ds_a = self._get_support(models[0], ds_layer, reverse=reverse)
                        ds_b = self._get_support(models[1], ds_layer, reverse=reverse)
                        mu_f_other = mu_f_list[other_pos] > 10 ** -8
                        nu_f_other = nu_f_list[other_pos] > 10 ** -8
                        ds_a, ds_b = self._align(
                            ds_a, ds_b, mu_f_other, nu_f_other,
                            kernels_backward[other_pos])
                        # Match the shape convention used for mu_for/mu_back:
                        # row count = current position's neuron count.
                        if ds_a.shape[0] == mu[0].shape[0]:
                            mu.append(ds_a)
                            nu.append(ds_b)
                mu_fuse, mu_iso, nu_fuse, nu_iso, k_for, k_back = self.get_mapping(mu, nu)
                kernels_forward[l+1] = k_for
                kernels_backward[l+1] = k_back
                mu_i_list[l+1] = mu_iso[:, None]
                nu_i_list[l+1] = nu_iso[:, None].copy()
                mu_f_list[l+1] = mu_fuse[:, None].copy()
                nu_f_list[l+1] = nu_fuse[:, None].copy()
                # enforce tied permutations across identity skip boundaries
                if l+1 in tied_to:
                    src = tied_to[l+1]
                    mu_f_list[l+1] = mu_f_list[src].copy()
                    mu_i_list[l+1] = mu_i_list[src].copy()
                    nu_f_list[l+1] = nu_f_list[src].copy()
                    nu_i_list[l+1] = nu_i_list[src].copy()
                    kernels_forward[l+1] = kernels_forward[src].copy()
                    kernels_backward[l+1] = kernels_backward[src].copy()
                # Force isolation: lock certain neurons as isolated
                if hasattr(self, '_force_isolated') and (l+1) in self._force_isolated:
                    force_mask = self._force_isolated[l+1]
                    fused_A = (mu_f_list[l+1] > self.eps).flatten()
                    fused_B = (nu_f_list[l+1] > self.eps).flatten()
                    dim = len(force_mask)
                    mu_f_list[l+1][force_mask] = 0
                    mu_i_list[l+1][force_mask] = 1.0 / dim
                    nu_f_list[l+1][force_mask] = 0
                    nu_i_list[l+1][force_mask] = 1.0 / dim
                    # Subset kernels to remaining fused neurons
                    new_fused_A = (mu_f_list[l+1] > self.eps).flatten()
                    new_fused_B = (nu_f_list[l+1] > self.eps).flatten()
                    keep_A = new_fused_A[fused_A]
                    keep_B = new_fused_B[fused_B]
                    if np.any(keep_A) and np.any(keep_B):
                        k_sub = kernels_forward[l+1][np.ix_(keep_B, keep_A)]
                        rs = k_sub.sum(axis=1, keepdims=True); rs[rs==0] = 1
                        kernels_forward[l+1] = k_sub / rs
                        kb_sub = kernels_backward[l+1][np.ix_(keep_A, keep_B)]
                        cs = kb_sub.sum(axis=0, keepdims=True); cs[cs==0] = 1
                        kernels_backward[l+1] = kb_sub / cs
            #s = 0
            #for l in range(L):
            #    x = self.get_support(models[0], layers1[l])
            #    y = self.get_support(models[1], layers2[l])
            #    y_b = kernels_forward[l + 1] @ y @ kernels_backward[l]
            #    s += np.linalg.norm(x - y_b)
            #print(i, s)
        self.kernel_forward = kernels_forward
        self.kernel_backward = kernels_backward
        return mu_f_list, nu_f_list, mu_i_list, nu_i_list, kernels_forward, kernels_backward


    def _compute_kernels_decoupled(self, models, layers1, layers2, L,
                                    mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                                    kernels_forward, kernels_backward):
        """
        Decoupled alignment: compute alignment (permutation) at alpha=0,
        then compute fused/isolated partition at target alpha, and use the
        reference permutation restricted to fused neurons.
        Guarantees monotonicity: higher alpha = more isolated neurons = more
        capacity, with the same (optimal) alignment for fused neurons.
        """
        eps = self.eps

        # Step 1: Run PCD at alpha=0 for reference alignment
        saved_sink = self.sink_weight
        saved_length = self.length
        self.sink_weight = [0.0]
        self.length = 1
        self.pos = 0

        ref_mu_f = [x.copy() for x in mu_f_list]
        ref_nu_f = [x.copy() for x in nu_f_list]
        ref_mu_i = [x.copy() for x in mu_i_list]
        ref_nu_i = [x.copy() for x in nu_i_list]
        ref_k_fwd = [x.copy() for x in kernels_forward]
        ref_k_bwd = [x.copy() for x in kernels_backward]

        ref_result = self._compute_kernels_pcd(
            models, layers1, layers2, L,
            ref_mu_f, ref_nu_f, ref_mu_i, ref_nu_i, ref_k_fwd, ref_k_bwd)
        ref_mu_f, ref_nu_f, ref_mu_i, ref_nu_i, ref_k_fwd, ref_k_bwd = ref_result

        # Step 2: Run PCD at target alpha for masks
        self.sink_weight = saved_sink
        self.length = saved_length
        self.pos = 0

        target_result = self._compute_kernels_pcd(
            models, layers1, layers2, L,
            mu_f_list, nu_f_list, mu_i_list, nu_i_list,
            kernels_forward, kernels_backward)
        mu_f_list, nu_f_list, mu_i_list, nu_i_list, kernels_forward, kernels_backward = target_result

        # Step 3: Replace target kernels with reference kernels restricted to fused neurons
        for l in range(len(kernels_forward)):
            if l == 0 or l >= len(mu_f_list):
                continue

            fused_A = np.array(mu_f_list[l] > eps).flatten()
            fused_B = np.array(nu_f_list[l] > eps).flatten()
            n_fused_A = np.sum(fused_A)
            n_fused_B = np.sum(fused_B)

            if n_fused_A == 0 or n_fused_B == 0:
                continue

            # Reference kernel covers ALL neurons (alpha=0 → all fused)
            ref_k = ref_k_fwd[l]
            ref_kb = ref_k_bwd[l]

            # Subset to neurons that are fused at target alpha
            k_sub = ref_k[np.ix_(fused_B, fused_A)]
            row_sums = k_sub.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            kernels_forward[l] = k_sub / row_sums

            kb_sub = ref_kb[np.ix_(fused_A, fused_B)]
            col_sums = kb_sub.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            kernels_backward[l] = kb_sub / col_sums

        self.kernel_forward = kernels_forward
        self.kernel_backward = kernels_backward
        return mu_f_list, nu_f_list, mu_i_list, nu_i_list, kernels_forward, kernels_backward

    def _compute_kernels_hierarchical(self, models, layers1, layers2, L,
                                      mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                                      kernels_forward, kernels_backward):
        """
        Hierarchical nesting: compute reference alignment at alpha=0, determine
        which neurons to isolate at target alpha, then RE-OPTIMIZE the permutation
        for the remaining fused neurons with the isolated neurons locked out.
        Combines the benefit of alpha-specific optimization (better accuracy)
        with nesting guarantees (monotonicity).
        """
        eps = self.eps

        # Step 1: Reference PCD at alpha=0
        saved_sink = self.sink_weight
        saved_length = self.length
        self.sink_weight = [0.0]
        self.length = 1
        self.pos = 0

        ref_mu_f = [x.copy() for x in mu_f_list]
        ref_nu_f = [x.copy() for x in nu_f_list]
        ref_mu_i = [x.copy() for x in mu_i_list]
        ref_nu_i = [x.copy() for x in nu_i_list]
        ref_k_fwd = [x.copy() for x in kernels_forward]
        ref_k_bwd = [x.copy() for x in kernels_backward]

        ref_result = self._compute_kernels_pcd(
            models, layers1, layers2, L,
            ref_mu_f, ref_nu_f, ref_mu_i, ref_nu_i, ref_k_fwd, ref_k_bwd)
        ref_mu_f, ref_nu_f, ref_mu_i, ref_nu_i, ref_k_fwd, ref_k_bwd = ref_result

        # Step 2: Target-alpha PCD to determine which neurons to isolate
        self.sink_weight = saved_sink
        self.length = saved_length
        self.pos = 0

        tgt_mu_f = [x.copy() for x in mu_f_list]
        tgt_nu_f = [x.copy() for x in nu_f_list]
        tgt_mu_i = [x.copy() for x in mu_i_list]
        tgt_nu_i = [x.copy() for x in nu_i_list]
        tgt_k_fwd = [x.copy() for x in kernels_forward]
        tgt_k_bwd = [x.copy() for x in kernels_backward]

        tgt_result = self._compute_kernels_pcd(
            models, layers1, layers2, L,
            tgt_mu_f, tgt_nu_f, tgt_mu_i, tgt_nu_i, tgt_k_fwd, tgt_k_bwd)
        tgt_mu_f, tgt_nu_f, tgt_mu_i, tgt_nu_i, _, _ = tgt_result

        # Step 3: Build force-isolation masks from target-alpha masks
        force_isolated = {}
        for l in range(1, len(tgt_mu_f)):
            isolated_A = (tgt_mu_i[l] > eps).flatten()
            if np.any(isolated_A):
                force_isolated[l] = isolated_A

        # Step 4: Re-optimize at alpha=0 with forced isolation
        # Initialize from reference kernels (good starting point)
        self.sink_weight = [0.0]
        self.length = 1
        self.pos = 0
        self._force_isolated = force_isolated

        reopt_mu_f = [x.copy() for x in ref_mu_f]
        reopt_nu_f = [x.copy() for x in ref_nu_f]
        reopt_mu_i = [x.copy() for x in ref_mu_i]
        reopt_nu_i = [x.copy() for x in ref_nu_i]
        reopt_k_fwd = [x.copy() for x in ref_k_fwd]
        reopt_k_bwd = [x.copy() for x in ref_k_bwd]

        result = self._compute_kernels_pcd(
            models, layers1, layers2, L,
            reopt_mu_f, reopt_nu_f, reopt_mu_i, reopt_nu_i, reopt_k_fwd, reopt_k_bwd)

        self._force_isolated = {}
        self.sink_weight = saved_sink
        self.length = saved_length

        self.kernel_forward = result[4]
        self.kernel_backward = result[5]
        return result

    def _compute_kernels_warmstart(self, models, layers1, layers2, L,
                                    mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                                    kernels_forward, kernels_backward):
        """
        Warm-started PCD: compute reference alignment at alpha=0, then
        run target-alpha PCD initialized from the reference (not identity).
        Combines robustness (good init) with optimality (alpha-specific adaptation).
        """
        # Step 1: Reference PCD at alpha=0
        saved_sink = self.sink_weight
        saved_length = self.length
        self.sink_weight = [0.0]
        self.length = 1
        self.pos = 0

        ref_mu_f = [x.copy() for x in mu_f_list]
        ref_nu_f = [x.copy() for x in nu_f_list]
        ref_mu_i = [x.copy() for x in mu_i_list]
        ref_nu_i = [x.copy() for x in nu_i_list]
        ref_k_fwd = [x.copy() for x in kernels_forward]
        ref_k_bwd = [x.copy() for x in kernels_backward]

        ref_result = self._compute_kernels_pcd(
            models, layers1, layers2, L,
            ref_mu_f, ref_nu_f, ref_mu_i, ref_nu_i, ref_k_fwd, ref_k_bwd)
        ref_mu_f, ref_nu_f, ref_mu_i, ref_nu_i, ref_k_fwd, ref_k_bwd = ref_result

        # Step 2: Target-alpha PCD initialized from reference
        self.sink_weight = saved_sink
        self.length = saved_length
        self.pos = 0

        result = self._compute_kernels_pcd(
            models, layers1, layers2, L,
            ref_mu_f, ref_nu_f, ref_mu_i, ref_nu_i,
            [x.copy() for x in ref_k_fwd], [x.copy() for x in ref_k_bwd])

        return result

    def _compute_kernels(self, models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list, kernels_forward,
                         kernels_backward, data=None, out_ens=False, reverse=False, pgd=False):
        if pgd and getattr(self, 'warmstart', False):
            return self._compute_kernels_warmstart(models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                                                    kernels_forward, kernels_backward)
        if pgd and self.hierarchical:
            return self._compute_kernels_hierarchical(models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                                                       kernels_forward, kernels_backward)
        if pgd and self.decoupled:
            return self._compute_kernels_decoupled(models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                                                    kernels_forward, kernels_backward)
        if pgd:
            return self._compute_kernels_pcd(models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                                             kernels_forward, kernels_backward)
        for l in range(L):
            w_a = models[0].get_incoming_weights(layers1[l], numpy=True)
            l1 = layers1[l]
            l2 = layers2[l]
            if data is not None and reverse and l!=L-1 and not self.combine_costs:
                l1 = layers1[l+1]
                l2 = layers2[l+1]
            if len(w_a.shape) == 2 and l1:
                mu = self._get_support(models[0], l1, data=data, reverse=reverse)
                nu = self._get_support(models[1], l2, data=data, reverse=reverse)
                if data is None:
                    mu_f = mu_f_list[-1] > 10 ** -8
                    nu_f = nu_f_list[-1] > 10 ** -8
                    mu, nu = self._align(mu, nu, mu_f, nu_f, kernels_backward[-1])

                if self.combine_costs:
                    mu_f = mu_f_list[-1] > 10 ** -8
                    nu_f = nu_f_list[-1] > 10 ** -8
                    mu_w = self._get_support(models[0], l1, reverse=reverse)
                    nu_w = self._get_support(models[1], l2, reverse=reverse)
                    mu_w, nu_w = self._align(mu_w, nu_w, mu_f, nu_f, kernels_backward[-1])
                    mu = [mu, mu_w]
                    nu = [nu, nu_w]
                else:
                    mu = [mu]
                    nu = [nu]
                mu_fuse, mu_iso, nu_fuse, nu_iso, k_for, k_back = self.get_mapping(mu, nu)
                if data is not None and reverse and l == L - 1:
                    mu = data.view(data.size(0), -1).T.numpy()
                    mu_fuse = np.ones(mu.shape[0]) / mu.shape[0]
                    nu_fuse = mu_fuse
                    mu_iso = np.zeros(mu.shape[0])
                    nu_iso = mu_iso
                mu_i_list.append(mu_iso[:, None].copy())
                nu_i_list.append(nu_iso[:, None].copy())
                mu_f_list.append(mu_fuse[:, None].copy())
                nu_f_list.append(nu_fuse[:, None].copy())

                if l < L - 1:
                    kernels_forward.append(k_for.copy())
                    kernels_backward.append(k_back.copy())
                else:
                    # case l == L-1:
                    if not reverse or data is None:
                        mu = self._get_support(models[0], layers1[L - 1], data=data, reverse=reverse)
                        nu = self._get_support(models[1], layers2[L - 1], data=data, reverse=reverse)
                    if out_ens:
                        kernels_forward.append(np.identity(0 * len(mu)))
                        kernels_backward.append(np.identity(2 * len(nu)))
                    else:
                        kernels_forward.append(np.identity(len(mu)))
                        kernels_backward.append(np.identity(len(nu)))
            elif len(w_a.shape) == 4: #CNNs
                mu = self._get_support(models[0], layers1[l], data=data, reverse=reverse)
                nu = self._get_support(models[1], layers2[l], data=data, reverse=reverse)
                # Transpose to (B, C, H, W)
                if data is not None:
                    mu = mu.transpose(3, 2, 1, 0)
                    nu = nu.transpose(3, 2, 1, 0)
                    # Reshape to (C, B * H * W)
                    mu = mu.reshape(mu.shape[0], mu.shape[1], -1).transpose(1, 0, 2).reshape(mu.shape[1], -1)
                    nu = nu.reshape(nu.shape[0], nu.shape[1], -1).transpose(1, 0, 2).reshape(nu.shape[1], -1)
                else:
                    mu_f = mu_f_list[-1] > 10 ** -8
                    nu_f = nu_f_list[-1] > 10 ** -8
                    ker_adj = kernels_backward[-1]
                    mu_f = np.array(mu_f).flatten().tolist()
                    nu_f = np.array(nu_f).flatten().tolist()
                    mu, nu = self._align(mu, nu, mu_f, nu_f, ker_adj)
                mu = [mu]
                nu = [nu]
                if self.combine_costs:
                    mu_w = self._get_support(models[0], layers1[l])
                    nu_w = self._get_support(models[1], layers2[l])
                    mu_w = mu_w.reshape(mu_w.shape[0], mu_w.shape[1] * mu_w.shape[2] * mu_w.shape[3])
                    nu_w = nu_w.reshape(nu_w.shape[0], nu_w.shape[1] * nu_w.shape[2] * nu_w.shape[3])
                    mu.append(mu_w)
                    nu.append(nu_w)
                mu_fuse, mu_iso, nu_fuse, nu_iso, k_for, k_back = self.get_mapping(mu, nu)
                mu_i_list.append(mu_iso[:, None].copy())
                nu_i_list.append(nu_iso[:, None].copy())
                mu_f_list.append(mu_fuse[:, None].copy())
                nu_f_list.append(nu_fuse[:, None].copy())

                if l < L - 1:
                    kernels_forward.append(k_for.copy())
                    kernels_backward.append(k_back.copy())
                else:
                    # case l == L-1:
                    mu = self._get_support(models[0], layers1[L - 1], data=data, reverse=reverse)
                    nu = self._get_support(models[1], layers2[L - 1], data=data, reverse=reverse)
                    if out_ens:
                        kernels_forward.append(np.identity(0*len(mu)))
                        kernels_backward.append(np.identity(2*len(nu)))
                    else:
                        kernels_forward.append(np.identity(len(mu)))
                        kernels_backward.append(np.identity(len(nu)))

        self.kernel_forward = kernels_forward
        self.kernel_backward = kernels_backward
        return mu_f_list, nu_f_list, mu_i_list, nu_i_list, kernels_forward, kernels_backward


    def _fuse_two_models_partial(self, models: [BaseModel], data=None, lambdas=None, out_ens=False, reverse=False, pgd=False):
        # partial fusion of two models with partial OT.
        # returns a priori a sequence of matrices W^l_i, for l indexing the layer and i=0, 1, 2, where i=0 is the
        # fused model part, i=1 is the isolated part of the first model and i=2 is the isolated part of the second model
        (models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list, kernels_forward, kernels_backward,
         lambdas, lambdas_length) = self._initialize_fusion(models, lambdas=lambdas, pgd=pgd)
        mu_f_list, nu_f_list, mu_i_list, nu_i_list, kernels_forward, kernels_backward \
            = self._compute_kernels(models, layers1, layers2, L, mu_f_list, nu_f_list, mu_i_list, nu_i_list,
                                    kernels_forward, kernels_backward, data=data, out_ens=out_ens, reverse=reverse, pgd=pgd)

        # Store OT alignment info for BatchNorm reconstruction
        self._mu_f_list = mu_f_list
        self._mu_i_list = mu_i_list
        self._nu_f_list = nu_f_list
        self._nu_i_list = nu_i_list
        self._kernels_forward = kernels_forward
        self._kernels_backward = kernels_backward

        # recompute as they might have changed with pgd
        layers1 = [layer for layer in models[0].get_layer_names_with_weights()
                   if layer not in models[0].get_residual_layers()[0]]
        layers2 = [layer for layer in models[1].get_layer_names_with_weights()
                   if layer not in models[1].get_residual_layers()[0]]
        L = len(layers1)
        A1_dict = {}
        A2_dict = {}
        A3_dict = {}
        # a counter for actual linear layers
        l = 0
        layer_to_kernel_idx = {}  # NEW: maps layer name -> kernel index for skip connections

        for l_all in range(L):
            alpha_l = lambdas[l % lambdas_length]
            self.lambdas_l.append(alpha_l)
            w_a = models[0].get_incoming_weights(layers1[l_all], numpy=True)
            w_b = models[1].get_incoming_weights(layers2[l_all], numpy=True)
            if reverse:
                w_a = w_a.T
                w_b = w_b.T
            mfel = mu_f_list[l] > self.eps
            mfelp = mu_f_list[l + 1] > self.eps
            miel = mu_i_list[l] > self.eps
            mielp = mu_i_list[l + 1] > self.eps
            nfel = nu_f_list[l] > self.eps
            nfelp = nu_f_list[l + 1] > self.eps
            niel = nu_i_list[l] > self.eps
            nielp = nu_i_list[l + 1] > self.eps

            # the below is to split weights in case neurons contribute to both the fused and isolated parts...
            # Note that we always handle such cases based on support for the layer where the weights are incoming.
            mwfp = mu_f_list[l + 1][mfelp, None].astype(float) / (mu_f_list[l + 1][mfelp, None].astype(float) + mu_i_list[l + 1][mfelp, None].astype(float))
            mwip = mu_i_list[l + 1][mielp, None].astype(float) / (mu_i_list[l + 1][mielp, None].astype(float) + mu_f_list[l + 1][mielp, None].astype(float))
            nwfp = nu_f_list[l + 1][nfelp, None].astype(float) / (nu_f_list[l + 1][nfelp, None].astype(float) + nu_i_list[l + 1][nfelp, None].astype(float))
            nwip = nu_i_list[l + 1][nielp, None].astype(float) / (nu_i_list[l + 1][nielp, None].astype(float) + nu_f_list[l + 1][nielp, None].astype(float))

            mfelp = np.array(mfelp).flatten()
            mielp = np.array(mielp).flatten()
            nfelp = np.array(nfelp).flatten()
            nielp = np.array(nielp).flatten()
            mfel = np.array(mfel).flatten()
            miel = np.array(miel).flatten()
            nfel = np.array(nfel).flatten()
            niel = np.array(niel).flatten()

            if len(w_a.shape) == 2:
                # fusion of linear layers
                # get relevant decompositions of the weight matrices: (ff is fuse-fuse, fi is fuse-isolated, etc.)
                if l_all > 0 and l_all < L - 1:

                    if mfel.shape[0] != w_a.shape[1]:
                        k = int(w_a.shape[1] / mfel.shape[0])
                        mfel = [x for x in mfel for _ in range(k)]
                        miel = [x for x in miel for _ in range(k)]
                        nfel = [x for x in nfel for _ in range(k)]
                        niel = [x for x in niel for _ in range(k)]
                    w_a_ff = w_a[mfelp, :][:, mfel] * mwfp
                    w_a_fi = w_a[mielp, :][:, mfel] * mwip
                    w_a_if = w_a[mfelp, :][:, miel] * mwfp
                    w_a_ii = w_a[mielp, :][:, miel] * mwip

                    w_b_ff = w_b[nfelp, :][:, nfel] * nwfp
                    w_b_fi = w_b[nielp, :][:, nfel] * nwip
                    w_b_if = w_b[nfelp, :][:, niel] * nwfp
                    w_b_ii = w_b[nielp, :][:, niel] * nwip
                elif l_all == L - 1:

                    w_a_ff = w_a[:, :][:, mfel]
                    w_a_fi = w_a[0:0, :][:, mfel]  # 0
                    w_a_if = w_a[:, :][:, miel]
                    w_a_ii = w_a[0:0, :][:, miel]  # 0

                    w_b_ff = w_b[:, :][:, nfel]
                    w_b_fi = w_b[0:0, :][:, nfel]  # 0
                    w_b_if = w_b[:, :][:, niel]
                    w_b_ii = w_b[0:0, :][:, niel]  # 0

                    if out_ens:
                        w_a_ff = w_a[0:0, :][:, mfel] # 0
                        w_a_fi = w_a[:, :][:, mfel]
                        w_a_if = w_a[0:0, :][:, miel] # 0
                        w_a_ii = w_a[:, :][:, miel]

                        w_b_ff = w_b[0:0, :][:, nfel] # 0
                        w_b_fi = w_b[:, :][:, nfel]
                        w_b_if = w_b[0:0, :][:, niel] # 0
                        w_b_ii = w_b[:, :][:, niel]
                else:  # l == 1
                    w_a_ff = w_a[mfelp, :] * mwfp
                    w_a_fi = w_a[mielp, :] * mwip
                    w_a_if = w_a[mfelp, 0:0] * mwfp
                    w_a_ii = w_a[mielp, 0:0] * mwip

                    w_b_ff = w_b[nfelp, :] * nwfp
                    w_b_fi = w_b[nielp, :] * nwip
                    w_b_if = w_b[nfelp, 0:0] * nwfp
                    w_b_ii = w_b[nielp, 0:0] * nwip


                ker_for = kernels_forward[l + 1]
                ker_back = kernels_backward[l]
                layer_to_kernel_idx[layers1[l_all]] = l  # NEW: track for skip connections
                l += 1

                if w_a_ff.shape[1] != ker_back.shape[0]:
                    k = int(w_a_ff.shape[1] / ker_back.shape[0])
                    n = ker_back.shape[0]
                    I = np.eye(k)
                    ker_back_extended = ker_back[:, :, None, None] * I[None, None, :, :]

                    # Rearrange to shape (n*k, n*k)
                    ker_back_extended = ker_back_extended.transpose(0, 2, 1, 3).reshape(n * k, n * k)
                    ker_back = ker_back_extended


                weighting_a = alpha_l[0]
                weighting_b = alpha_l[1]
                if self.fix_mu:
                    A3 = np.concatenate([w_b_if, w_b_ii], axis=0)
                else:
                    A3 = np.concatenate([weighting_b * w_b_if, w_b_ii], axis=0)
                A2 = np.concatenate([ker_for @ (weighting_a * w_a_if), w_a_ii], axis=0)
                A1 = np.concatenate(
                    [weighting_b * w_b_ff + ker_for @ (weighting_a * w_a_ff) @ ker_back,
                     w_a_fi @ ker_back, w_b_fi], axis=0)

                if reverse:
                    A1 = A1.T
                    A2 = A2.T
                    A3 = A3.T

                A1_dict[layers1[l_all]] = A1
                A2_dict[layers1[l_all]] = A2
                A3_dict[layers1[l_all]] = A3

                self.mfels.append(mfel)
                self.miels.append(miel)
                self.nfels.append(nfel)
                self.niels.append(niel)

            elif len(w_a.shape) == 4:
                # fusion of convolutional layers (O, I, H, W)
                if l_all > 0 and l_all < L - 1:
                    w_a_ff = w_a[mfelp][:, mfel] * mwfp[:, None, None]
                    w_a_fi = w_a[mielp][:, mfel] * mwip[:, None, None]
                    w_a_if = w_a[mfelp][:, miel] * mwfp[:, None, None]
                    w_a_ii = w_a[mielp][:, miel] * mwip[:, None, None]

                    w_b_ff = w_b[nfelp][:, nfel] * nwfp[:, None, None]
                    w_b_fi = w_b[nielp][:, nfel] * nwip[:, None, None]
                    w_b_if = w_b[nfelp][:, niel] * nwfp[:, None, None]
                    w_b_ii = w_b[nielp][:, niel] * nwip[:, None, None]

                elif l_all == L - 1:
                    w_a_ff = w_a[:, mfel]
                    w_a_fi = w_a[0:0, mfel]
                    w_a_if = w_a[:, miel]
                    w_a_ii = w_a[0:0, miel]

                    w_b_ff = w_b[:, nfel]
                    w_b_fi = w_b[0:0, nfel]
                    w_b_if = w_b[:, niel]
                    w_b_ii = w_b[0:0, niel]

                else:  # l == 0
                    w_a_ff = w_a[mfelp] * mwfp[:, None, None]
                    w_a_fi = w_a[mielp] * mwip[:, None, None]
                    w_a_if = w_a[mfelp, 0:0] * mwfp[:, None, None]
                    w_a_ii = w_a[mielp, 0:0] * mwip[:, None, None]

                    w_b_ff = w_b[nfelp] * nwfp[:, None, None]
                    w_b_fi = w_b[nielp] * nwip[:, None, None]
                    w_b_if = w_b[nfelp, 0:0] * nwfp[:, None, None]
                    w_b_ii = w_b[nielp, 0:0] * nwip[:, None, None]

                ker_for = kernels_forward[l + 1]
                ker_back = kernels_backward[l]
                layer_to_kernel_idx[layers1[l_all]] = l  # NEW: track for skip connections
                l += 1
                if self.fix_mu:
                    A3 = np.concatenate([w_b_if, w_b_ii], axis=0)
                else:
                    A3 = np.concatenate([alpha_l[1] * w_b_if, w_b_ii], axis=0)
                A2 = np.concatenate([self._transform_conv_weights(alpha_l[0] * w_a_if, ker_for, np.eye(w_a_if.shape[1])), w_a_ii], axis=0)
                A1 = np.concatenate([
                    alpha_l[1] * w_b_ff +
                    self._transform_conv_weights(alpha_l[0] * w_a_ff, ker_for, ker_back),
                    self._transform_conv_weights(w_a_fi, np.eye(w_a_fi.shape[0]), ker_back),
                    w_b_fi
                ], axis=0)

                A1_dict[layers1[l_all]] = A1
                A2_dict[layers1[l_all]] = A2
                A3_dict[layers1[l_all]] = A3

                self.mfels.append(mfel)
                self.miels.append(miel)
                self.nfels.append(nfel)
                self.niels.append(niel)

        # =================================================================
        # Fuse residual skip connection convolutions
        # =================================================================
        # The skip conv maps from the block's input space to its output
        # space.  We decompose it into fused/isolated parts using the
        # same OT masks and kernels as the main-branch convolutions,
        # exactly mirroring the treatment of regular conv layers above.
        res_info = models[0].get_residual_layers()
        residual_map = res_info[2] if len(res_info) > 2 else {}

        for ds_layer, (first_conv, last_conv) in residual_map.items():
            if first_conv not in layer_to_kernel_idx or last_conv not in layer_to_kernel_idx:
                continue

            l_in = layer_to_kernel_idx[first_conv]
            l_out = layer_to_kernel_idx[last_conv] + 1

            if l_in >= len(kernels_backward) or l_out >= len(kernels_forward):
                w_a = models[0].get_incoming_weights(ds_layer, numpy=True)
                w_b = models[1].get_incoming_weights(ds_layer, numpy=True)
                alpha_l = lambdas[0] if lambdas_length == 1 else lambdas[l_in % lambdas_length]
                A1 = alpha_l[1] * w_b + alpha_l[0] * w_a
                A2 = np.zeros((0, 0) + w_a.shape[2:], dtype=w_a.dtype)
                A3 = np.zeros((0, 0) + w_b.shape[2:], dtype=w_b.dtype)
                A1_dict[ds_layer] = A1
                A2_dict[ds_layer] = A2
                A3_dict[ds_layer] = A3
                continue

            w_a = models[0].get_incoming_weights(ds_layer, numpy=True)
            w_b = models[1].get_incoming_weights(ds_layer, numpy=True)

            alpha_l = lambdas[0] if lambdas_length == 1 else lambdas[l_in % lambdas_length]
            weighting_a = alpha_l[0]
            weighting_b = alpha_l[1]

            # Input-space masks  (same space as first_conv's input)
            mfel  = np.array(mu_f_list[l_in] > self.eps).flatten()
            miel  = np.array(mu_i_list[l_in] > self.eps).flatten()
            nfel  = np.array(nu_f_list[l_in] > self.eps).flatten()
            niel  = np.array(nu_i_list[l_in] > self.eps).flatten()

            # Output-space masks (same space as last_conv's output)
            mfelp = mu_f_list[l_out] > self.eps
            mielp = mu_i_list[l_out] > self.eps
            nfelp = nu_f_list[l_out] > self.eps
            nielp = nu_i_list[l_out] > self.eps

            # Output weight splits (same logic as main-branch conv layers)
            mwfp = mu_f_list[l_out][mfelp, None].astype(float) / (
                mu_f_list[l_out][mfelp, None].astype(float) + mu_i_list[l_out][mfelp, None].astype(float))
            mwip = mu_i_list[l_out][mielp, None].astype(float) / (
                mu_i_list[l_out][mielp, None].astype(float) + mu_f_list[l_out][mielp, None].astype(float))
            nwfp = nu_f_list[l_out][nfelp, None].astype(float) / (
                nu_f_list[l_out][nfelp, None].astype(float) + nu_i_list[l_out][nfelp, None].astype(float))
            nwip = nu_i_list[l_out][nielp, None].astype(float) / (
                nu_i_list[l_out][nielp, None].astype(float) + nu_f_list[l_out][nielp, None].astype(float))

            mfelp = np.array(mfelp).flatten()
            mielp = np.array(mielp).flatten()
            nfelp = np.array(nfelp).flatten()
            nielp = np.array(nielp).flatten()

            # Decompose weights (mirrors the regular conv-layer treatment)
            w_a_ff = w_a[mfelp][:, mfel] * mwfp[:, None, None]
            w_a_fi = w_a[mielp][:, mfel] * mwip[:, None, None]
            w_a_if = w_a[mfelp][:, miel] * mwfp[:, None, None]
            w_a_ii = w_a[mielp][:, miel] * mwip[:, None, None]

            w_b_ff = w_b[nfelp][:, nfel] * nwfp[:, None, None]
            w_b_fi = w_b[nielp][:, nfel] * nwip[:, None, None]
            w_b_if = w_b[nfelp][:, niel] * nwfp[:, None, None]
            w_b_ii = w_b[nielp][:, niel] * nwip[:, None, None]

            ker_for = kernels_forward[l_out]
            ker_back = kernels_backward[l_in]

            if self.fix_mu:
                A3 = np.concatenate([w_b_if, w_b_ii], axis=0)
            else:
                A3 = np.concatenate([weighting_b * w_b_if, w_b_ii], axis=0)
            A2 = np.concatenate([
                self._transform_conv_weights(weighting_a * w_a_if, ker_for, np.eye(w_a_if.shape[1])),
                w_a_ii], axis=0)
            A1 = np.concatenate([
                weighting_b * w_b_ff +
                self._transform_conv_weights(weighting_a * w_a_ff, ker_for, ker_back),
                self._transform_conv_weights(w_a_fi, np.eye(w_a_fi.shape[0]), ker_back),
                w_b_fi
            ], axis=0)

            A1_dict[ds_layer] = A1
            A2_dict[ds_layer] = A2
            A3_dict[ds_layer] = A3

        # =================================================================
        # Fuse identity residual connections (non-downsample blocks)
        # =================================================================
        # For blocks without downsample, the original skip is identity.
        # After partial fusion the input/output spaces may differ in size
        # and ordering, so we compute a proper skip projection.
        identity_residual_map = res_info[4] if len(res_info) > 4 else {}

        for skip_name, (first_conv, last_conv) in identity_residual_map.items():
            if first_conv not in layer_to_kernel_idx or last_conv not in layer_to_kernel_idx:
                continue

            l_in = layer_to_kernel_idx[first_conv]
            l_out = layer_to_kernel_idx[last_conv] + 1

            if l_in >= len(kernels_backward) or l_out >= len(kernels_forward):
                continue

            alpha_l = lambdas[0] if lambdas_length == 1 else lambdas[l_in % lambdas_length]
            weighting_a = alpha_l[0]
            weighting_b = alpha_l[1]

            if self.direct_skip_composition:
                # Instead of OT-decomposing the identity matrix (which creates
                # cross-terms when fused/isolated partitions differ at input vs
                # output), compose the permutation kernels directly.
                ker_for = kernels_forward[l_out]
                ker_back = kernels_backward[l_in]
                n_fused = ker_for.shape[0]

                n_iso_A_in = int(np.sum(mu_i_list[l_in] > self.eps))
                n_iso_B_in = int(np.sum(nu_i_list[l_in] > self.eps))
                n_iso_A_out = int(np.sum(mu_i_list[l_out] > self.eps))
                n_iso_B_out = int(np.sum(nu_i_list[l_out] > self.eps))

                total_in = n_fused + n_iso_A_in + n_iso_B_in
                total_out = n_fused + n_iso_A_out + n_iso_B_out

                W = np.zeros((total_out, total_in, 1, 1), dtype=np.float32)
                # Fused block: compose permutations from input to output space
                W[:n_fused, :n_fused, 0, 0] = (
                    weighting_a * (ker_for @ ker_back) +
                    weighting_b * np.eye(n_fused)
                ).astype(np.float32)
                # Isolated A: identity (each model's isolated neurons pass through)
                n_iso_A = min(n_iso_A_in, n_iso_A_out)
                if n_iso_A > 0:
                    W[n_fused:n_fused+n_iso_A, n_fused:n_fused+n_iso_A, 0, 0] = np.eye(n_iso_A, dtype=np.float32)
                # Isolated B: identity
                n_iso_B = min(n_iso_B_in, n_iso_B_out)
                if n_iso_B > 0:
                    off_out = n_fused + n_iso_A_out
                    off_in = n_fused + n_iso_A_in
                    W[off_out:off_out+n_iso_B, off_in:off_in+n_iso_B, 0, 0] = np.eye(n_iso_B, dtype=np.float32)

                A1_dict[skip_name] = W
                A2_dict[skip_name] = np.zeros((0, 0, 1, 1), dtype=np.float32)
                A3_dict[skip_name] = np.zeros((0, 0, 1, 1), dtype=np.float32)
            else:
                # Original: OT-decompose the identity matrix
                c = models[0].get_incoming_weights(first_conv, numpy=True).shape[1]
                w_a = np.eye(c, dtype=np.float32).reshape(c, c, 1, 1)
                w_b = np.eye(c, dtype=np.float32).reshape(c, c, 1, 1)

                # Input-space masks
                mfel  = np.array(mu_f_list[l_in] > self.eps).flatten()
                miel  = np.array(mu_i_list[l_in] > self.eps).flatten()
                nfel  = np.array(nu_f_list[l_in] > self.eps).flatten()
                niel  = np.array(nu_i_list[l_in] > self.eps).flatten()

                # Output-space masks
                mfelp = mu_f_list[l_out] > self.eps
                mielp = mu_i_list[l_out] > self.eps
                nfelp = nu_f_list[l_out] > self.eps
                nielp = nu_i_list[l_out] > self.eps

                mwfp = mu_f_list[l_out][mfelp, None].astype(float) / (
                    mu_f_list[l_out][mfelp, None].astype(float) + mu_i_list[l_out][mfelp, None].astype(float))
                mwip = mu_i_list[l_out][mielp, None].astype(float) / (
                    mu_i_list[l_out][mielp, None].astype(float) + mu_f_list[l_out][mielp, None].astype(float))
                nwfp = nu_f_list[l_out][nfelp, None].astype(float) / (
                    nu_f_list[l_out][nfelp, None].astype(float) + nu_i_list[l_out][nfelp, None].astype(float))
                nwip = nu_i_list[l_out][nielp, None].astype(float) / (
                    nu_i_list[l_out][nielp, None].astype(float) + nu_f_list[l_out][nielp, None].astype(float))

                mfelp = np.array(mfelp).flatten()
                mielp = np.array(mielp).flatten()
                nfelp = np.array(nfelp).flatten()
                nielp = np.array(nielp).flatten()

                w_a_ff = w_a[mfelp][:, mfel] * mwfp[:, None, None]
                w_a_fi = w_a[mielp][:, mfel] * mwip[:, None, None]
                w_a_if = w_a[mfelp][:, miel] * mwfp[:, None, None]
                w_a_ii = w_a[mielp][:, miel] * mwip[:, None, None]

                w_b_ff = w_b[nfelp][:, nfel] * nwfp[:, None, None]
                w_b_fi = w_b[nielp][:, nfel] * nwip[:, None, None]
                w_b_if = w_b[nfelp][:, niel] * nwfp[:, None, None]
                w_b_ii = w_b[nielp][:, niel] * nwip[:, None, None]

                ker_for = kernels_forward[l_out]
                ker_back = kernels_backward[l_in]

                if self.fix_mu:
                    A3 = np.concatenate([w_b_if, w_b_ii], axis=0)
                else:
                    A3 = np.concatenate([weighting_b * w_b_if, w_b_ii], axis=0)
                A2 = np.concatenate([
                    self._transform_conv_weights(weighting_a * w_a_if, ker_for, np.eye(w_a_if.shape[1])),
                    w_a_ii], axis=0)
                A1 = np.concatenate([
                    weighting_b * w_b_ff +
                    self._transform_conv_weights(weighting_a * w_a_ff, ker_for, ker_back),
                    self._transform_conv_weights(w_a_fi, np.eye(w_a_fi.shape[0]), ker_back),
                    w_b_fi
                ], axis=0)

                A1_dict[skip_name] = A1
                A2_dict[skip_name] = A2
                A3_dict[skip_name] = A3

        self._layer_to_kernel_idx = layer_to_kernel_idx
        return A1_dict, A2_dict, A3_dict

    def _transform_conv_weights(self, w, ker_for, ker_back):
        tmp = np.einsum('bchw,cd->bdhw', w, ker_back)
        out = np.tensordot(ker_for, tmp, axes=([1], [0]))  # (out_new, h, w, in_new)
        return out


    def _get_support(self, model, layer_name, data=None, reverse=False):
        if data is not None:
            activations = model.get_activations(layer_name, data, numpy=True).T
            if self.act:
                func = model.get_next_activation(layer_name, numpy=True)
                if func is not None:
                    activations = func(activations)
            # shape: (neurons of layer by datapoints)
            return activations
        else:
            weights = model.get_incoming_weights(layer_name, numpy=True)
            if reverse:
                axes = list(range(weights.ndim))
                axes[0], axes[1] = axes[1], axes[0]
                weights = weights.transpose(axes)
            # BN-adjusted: scale output neurons by gamma/sqrt(var+eps)
            if getattr(self, 'bn_adjusted_cost', False) and not reverse:
                import torch.nn as nn
                norm_map = {}
                for name, _ in model.named_modules():
                    if name.endswith('_conv1') or name.endswith('_conv2'):
                        bn_name = name.replace('_conv', '_bn')
                        if hasattr(model, bn_name):
                            norm_map[name] = bn_name
                    elif name == 'conv1' and hasattr(model, 'bn1'):
                        norm_map['conv1'] = 'bn1'
                if layer_name in norm_map:
                    bn = model.get_layer_by_name(norm_map[layer_name])
                    if bn is not None and hasattr(bn, 'weight') and bn.weight is not None:
                        gamma = bn.weight.detach().cpu().numpy()
                        var = bn.running_var.detach().cpu().numpy()
                        scale = gamma / np.sqrt(var + 1e-5)
                        if weights.ndim == 4:
                            weights = weights * scale[:, None, None, None]
                        elif weights.ndim == 2:
                            weights = weights * scale[:, None]
            # shape (for MLP): (neurons current layer by neurons previous layer)
            return weights

    def get_mapping(self, mu, nu):
        raise NotImplementedError
