import sys
from .base_fusion import BaseFusion
import numpy as np
import ot

sys.path.append("../")


class PartialFusion(BaseFusion):
    def __init__(self, eps=10**-8, alphas=None, c_pointsink=0, c_sinksink=10 ** 7, reg=0, act=False,
                 uniform=True, fix_mu=False, combine_costs=False, pgd=False,
                 tied_permutations=False, direct_skip_composition=False,
                 soft_tie_downsample=False,
                 decoupled=False, hierarchical=False, warmstart=False, bn_adjusted_cost=False):
        super().__init__(eps=eps, act=act, fix_mu=fix_mu, combine_costs=combine_costs, pgd=pgd,
                         tied_permutations=tied_permutations, direct_skip_composition=direct_skip_composition,
                         soft_tie_downsample=soft_tie_downsample)
        self.decoupled = decoupled
        self.hierarchical = hierarchical
        self.warmstart = warmstart
        self.bn_adjusted_cost = bn_adjusted_cost
        self.sink_weight = alphas  # the "sink" is the part of the distribution which lets you move mass into nothing
        # in particular, if sink_weight=0, then we have normal OT. if sink_weight=1, you only have the sink.
        self.c_pointsink = c_pointsink  # this gives the cost between points in the original distributions and the sink
        self.c_sinksink = c_sinksink  # this gives the cost between the sinks of both distributions
        # Overall, if c_pointsink=0 and c_sinksink=inf, then as much mass as possible is moved from normal points to sinks
        # Something like sink_weight=0.999 and c_pointsink=c_sinksink=delta would instead mean that only mass
        # between points is transported with cost lower than delta. The rest is put into the sink.
        self.reg = reg
        self.pos = 0
        if self.sink_weight is not None and not isinstance(self.sink_weight, list):
            self.sink_weight = [self.sink_weight]
            self.length = len(self.sink_weight)
        elif self.sink_weight is not None:
            self.length = len(self.sink_weight)
        self.uniform = uniform
        self.fix_mu = fix_mu


    def get_mapping(self, mu, nu):
        _, opt = self.get_similarity(mu, nu)
        mu = mu[0]
        nu = nu[0]
        self.pos += 1
        pi = opt['pi']  # coupling between the parts of mu and nu which are matched, so a coupling between mu_fuse and nu_fuse below
        pi_s = opt['pi_s']
        mu_fuse = np.sum(pi, axis=1)  # part of model 0 which is matched with model 1
        nu_fuse = np.sum(pi, axis=0)  # part of model 1 which his matched with model 0
        mu_iso = np.sum(pi_s[:mu.shape[0], :], axis=1) - mu_fuse  # part of model 0 which is not matched
        nu_iso = np.sum(pi_s[:, :nu.shape[0]], axis=0) - nu_fuse  # part of model 1 which is not matched
        k_for = np.transpose(pi[mu_fuse > self.eps, :][:, nu_fuse > self.eps] / mu_fuse[
            mu_fuse > self.eps, None])  # kernel forward resulting from pi, so pi = mu_fuse times k_for; restricted to relevant elements
        k_back = pi[mu_fuse > self.eps, :][:, nu_fuse > self.eps] / nu_fuse[
            None, nu_fuse > self.eps]  # kernel backward resulting from pi, so pi = nu_fuse times k_back; restricted to relevant elements
        if self.fix_mu:
            return mu_fuse, np.zeros_like(mu_iso), nu_fuse, nu_iso, k_for, k_back
        return mu_fuse, mu_iso, nu_fuse, nu_iso, k_for, k_back

    def get_similarity(self, x: np.ndarray, y: np.ndarray) -> (float, dict):
        """
        This function assumes that x and y are the neurons of layers with x[i] being the support based
        on weights or activations. We also assume the previous layer was already aligned
        :param x:
        :param y:
        :return:
        """

        # Normal OT measures and cost
        if self.uniform:
            w_mu = np.ones(x[0].shape[0]) / x[0].shape[0]
            w_nu = np.ones(y[0].shape[0]) / y[0].shape[0]
        else:
            w_mu = np.sum(x[0], axis=1) / np.sum(x[0])
            w_nu = np.sum(y[0], axis=1) / np.sum(y[0])
        if self.combine_costs:
            cs = []
            for x_mu, x_nu in zip(x, y):
                c_i = ot.dist(x_mu, x_nu, metric='sqeuclidean')
                c_i = c_i / np.sum(c_i + 10**-8)
                cs.append(c_i * 1000)
            cs = np.stack(cs)  # shape (N, m, n)
            c = np.mean(cs, axis=0)
        else:
            c = ot.dist(x[0], y[0], metric='sqeuclidean')
        max_val = np.max(c)

        # Measures and cost including the sink
        sink_w = self.sink_weight[self.pos % self.length]
        sink_w = sink_w / (1+sink_w)

        w_mu_s = np.append(w_mu*(1-sink_w), sink_w)
        w_nu_s = np.append(w_nu*(1-sink_w), sink_w)
        n, m = c.shape
        c_s = np.zeros([n+1, m+1])
        c_s[:n, :m] = c
        c_s[n, :m] = self.c_pointsink
        c_s[:n, m] = self.c_pointsink
        c_s[n, m] = max_val + 1 # self.c_sinksink

        # Solve the partial OT problem:
        pi_s = ot.emd(w_mu_s, w_nu_s, c_s)
        if self.reg > 0:
            pi_s = ot.sinkhorn(w_mu_s, w_mu_s, c_s, reg=self.reg)

        sink_sink_w = pi_s[n, m]
        point_sink_w = sink_w - sink_sink_w  # this is the amount of mass which was put into the sink
        pi = pi_s[:n, :m]  # note that this is NOT a coupling between mu and nu. Only a "sub-coupling".
        ot_val_s = (pi_s * c_s).sum()
        if np.sum(pi) > 0:
            ot_val = (pi * c).sum() * (1/np.sum(pi))  # average cost of transportation among the mass which is tranported
        else:
            ot_val = 0
        return_dict = {'ot_val': ot_val, 'ot_val_s': ot_val_s, 'pi': pi, 'pi_s': pi_s, 'weight_sink': point_sink_w}
        return ot_val, return_dict
