import sys
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

from .base_fusion import BaseFusion

sys.path.append("../")


class GitRebasin(BaseFusion):
    """
    Git Re-Basin (Ainsworth, Hayase, Srinivasa 2023) — exact permutation
    alignment via the Hungarian algorithm.

    Drop-in replacement for PartialFusion when you want the same per-layer
    feature/cost construction (squared-Euclidean, optionally averaged across
    blocks via ``combine_costs``) and the same coordinate-descent style
    iterative refinement (``pgd=True``), but with a *full permutation* in
    place of a partial-OT coupling. Every neuron of A is matched to exactly
    one neuron of B; there is no sink, no isolated neurons, and no ``alpha``
    knob.

    With ``combine_costs=True, pgd=True`` this exactly reproduces the
    weight-matching variant of Git Re-Basin: the cost at each layer is the
    sum of squared distances between the incoming-weight rows and the
    outgoing-weight columns of A and B (after permuting neighbouring layers
    using their current best permutations), and the optimal permutation at
    each layer is solved by linear assignment.
    """

    def __init__(self, eps=10 ** -8, act=False, combine_costs=True, pgd=True,
                 tied_permutations=False, direct_skip_composition=False,
                 soft_tie_downsample=False):
        super().__init__(
            eps=eps, act=act, combine_costs=combine_costs, pgd=pgd,
            tied_permutations=tied_permutations,
            direct_skip_composition=direct_skip_composition,
            soft_tie_downsample=soft_tie_downsample,
        )

    # ------------------------------------------------------------------
    # Cost matrix — replicates the PartialFusion construction exactly so
    # that the only methodological difference is the Hungarian assignment.
    # ------------------------------------------------------------------
    def _build_cost(self, mu, nu):
        if self.combine_costs:
            cs = []
            for x_mu, x_nu in zip(mu, nu):
                c_i = ot.dist(x_mu, x_nu, metric='sqeuclidean')
                c_i = c_i / (np.sum(c_i) + 10 ** -8)
                cs.append(c_i * 1000)
            return np.mean(np.stack(cs), axis=0)
        return ot.dist(mu[0], nu[0], metric='sqeuclidean')

    def get_mapping(self, mu, nu):
        n_a = mu[0].shape[0]
        n_b = nu[0].shape[0]

        c = self._build_cost(mu, nu)
        row_ind, col_ind = linear_sum_assignment(c)
        n_match = len(row_ind)

        # Build the full permutation matrix in PartialFusion's (A, B)
        # convention:
        #   P[i, j] = 1  iff  neuron i of A is matched to neuron j of B
        # so  k_back = P  and  k_for = P^T.
        P_full = np.zeros((n_a, n_b))
        P_full[row_ind, col_ind] = 1.0

        # Matched neurons receive uniform mass; unmatched ones (only possible
        # in the rectangular case n_a != n_b) get zero mass and are dropped
        # from the kernels by the boolean-mask restriction below — this
        # mirrors how PartialFusion handles fused-vs-isolated separation.
        mu_fuse = np.zeros(n_a)
        nu_fuse = np.zeros(n_b)
        if n_match > 0:
            mu_fuse[row_ind] = 1.0 / n_match
            nu_fuse[col_ind] = 1.0 / n_match

        # PartialFusion stores kernels restricted to the fused subset using
        # boolean masks (preserving original index order, *not* matching
        # order). Reproduce that exactly so downstream code (kernel
        # composition, BN rebuild, etc.) sees identical conventions.
        fused_A = mu_fuse > self.eps
        fused_B = nu_fuse > self.eps
        k_back = P_full[fused_A, :][:, fused_B]
        k_for = P_full.T[fused_B, :][:, fused_A]

        mu_iso = np.zeros(n_a)
        nu_iso = np.zeros(n_b)
        return mu_fuse, mu_iso, nu_fuse, nu_iso, k_for, k_back

    # get_similarity is not needed by the fusion pipeline for this class,
    # but exposing it keeps the interface symmetric with PartialFusion
    # (useful if anyone wants to inspect the chosen assignment cost).
    def get_similarity(self, x, y):
        c = self._build_cost(x, y)
        row_ind, col_ind = linear_sum_assignment(c)
        ot_val = c[row_ind, col_ind].sum() / max(len(row_ind), 1)
        return ot_val, {
            'ot_val': ot_val,
            'row_ind': row_ind,
            'col_ind': col_ind,
            'cost': c[row_ind, col_ind],
        }
