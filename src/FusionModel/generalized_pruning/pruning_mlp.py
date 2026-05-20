import numpy as np
from sklearn.cluster import KMeans
import ot
import sys
sys.path.append("../../../")
from src.MLP import Deep_MLP, MLP
from src.FusionModel.generalized_pruning import BaseGP, WeightHierarchical, StochHierarchical
import torch
import warnings
from sklearn.cluster import KMeans



def prune_net_clean(net, smaller_sizes, t_dat, weight_based=False, both=False, weight_double=False, importance=None,
                    pi_forced_uniform=True, post_act_act=False, pre_cluster=False, use_refine=True, use_rand_iter=True, which_act=0, use_reg=None, clustering_algo='stoch', tol=1e-15):
    layers1 = net.get_layer_names_with_weights()
    L = len(layers1)
    act_list = []
    quant_list = []
    pi_list = []
    for l in range(L-1): # only until L-2 as L-1 is just the output layer, where we have no compression and just the identity kernel
        if not pre_cluster:
            layer_name = layers1[l]
            if not weight_based:
                activations = net.get_activations(layer_name, t_dat, numpy=True).T  # activations pre-activaiton function; but activation is applied elow with func
                activations /= np.linalg.norm(activations)
            else:
                activations = net.get_incoming_weights(layer_name, numpy=True)
                activations /= np.linalg.norm(activations)
                if weight_double:
                    acti2 = net.get_incoming_weights(layers1[l+1], numpy=True).T
                    acti2 /= np.linalg.norm(acti2)
                    activations = np.concatenate((activations, acti2), axis=1)
            len_act, _ = activations.shape
            func = net.get_next_activation(layer_name, numpy=True)
            if (func is not None) and (not weight_based):
                activations = func(activations)

            if both:
                acti2 = net.get_incoming_weights(layer_name, numpy=True)
                acti2 /= np.linalg.norm(acti2)
                activations = np.concatenate((activations, acti2), axis=1)
                if weight_double:
                    acti3 = net.get_incoming_weights(layers1[l + 1], numpy=True).T
                    acti3 /= np.linalg.norm(acti3)
                    activations = np.concatenate((activations, acti3), axis=1)

            act_list.append(activations)
            if use_reg:
                avg_cluster_size = len(activations)/smaller_sizes[l]
                max_size = np.ceil(2 * avg_cluster_size)
            else:
                max_size = None

            if importance is not None:
                weights = importance[l]
            else:
                weights = None

            if clustering_algo == 'stoch':
                centroids, labels, inertia = StochHierarchical(max_size, weights=weights).clustering(activations, smaller_sizes[l])
            elif clustering_algo == 'stoch_high':
                centroids, labels, inertia = StochHierarchical(max_size, weights=weights, n_restarts=5000).clustering(activations, smaller_sizes[l])
            elif clustering_algo == 'stoch_very_high':
                centroids, labels, inertia = StochHierarchical(max_size, weights=weights, n_restarts=50000).clustering(activations, smaller_sizes[l])
            elif clustering_algo == 'kmeans':
                kmeans = KMeans(n_clusters=smaller_sizes[l], n_init=10**5, max_iter=10**4, tol=10**-5).fit(activations)
                centroids, labels, inertia = kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_
            else:
                centroids, labels, inertia = WeightHierarchical().clustering(activations, smaller_sizes[l])
            print('Inertia', inertia)


            weights = np.array([np.sum(labels==i) for i in range(smaller_sizes[l])])/len_act
            quant_list.append([centroids, weights])

            # Below: calculate pi according to clustering
            pi = np.zeros([len_act, smaller_sizes[l]])
            for ind_cl in range(int(smaller_sizes[l])):
                pi[:, ind_cl] = (labels==ind_cl)/len_act
            pi_list.append(pi)

        else:
            w_a = net.get_incoming_weights(layers1[l], numpy=True)
            xss_in = w_a.copy()
            xss_out = net.get_incoming_weights(layers1[l+1], numpy=True).T.copy()
            xss = xss_in
            n_w, m_w = w_a.shape
            if l == 0:
                inds_in = np.arange(0, m_w)
            else:
                inds_in = inds_out.copy()
            if l == L - 1:
                inds_out = np.arange(0, n_w)
            else:
                norm_h = np.linalg.norm(w_a[:, inds_in], axis=1)
                sort_inds = np.argsort(norm_h)[::-1]
                inds_out = sort_inds[:smaller_sizes[l]]
                # inds_out = np.random.choice(np.arange(0,n_w),size=smaller_sizes[l], replace=False)

            n_ss = xss.shape[0]
            if use_rand_iter:
                N_iter = 1000
            else:
                N_iter = 1
            cur_val = np.inf
            for n_it in range(N_iter):
                w_centers = np.ones(smaller_sizes[l])/smaller_sizes[l]
                if use_rand_iter:
                    inds_out = np.random.choice(np.arange(0, xss.shape[0]), size=smaller_sizes[l], replace=False)
                    x_centers = xss[inds_out, :]
                else:
                    x_centers = xss[inds_out, :]
                w_full = np.ones(n_ss)/n_ss
                x_full = xss
                c = ot.dist(x_full, x_centers, metric='sqeuclidean')
                pi_t = ot.emd(w_full, w_centers, c)
                if use_refine:
                    ct = np.inf
                    while ct > np.sum(pi_t * c):
                        ct = np.sum(pi_t * c)
                        K_h = pi_t / np.sum(pi_t, axis=0)[None, :]
                        # new_centers = np.sum(K_h[:, :, None] * x_full[:, None, :], axis=0)
                        new_centers = K_h.T @ x_full
                        # new_centers = np.transpose(pi_t) @ x_full
                        c = ot.dist(x_full, new_centers, metric='sqeuclidean')
                        pi_t = ot.emd(w_full, w_centers, c)
                if np.sum(c * pi_t) < cur_val:
                    print(cur_val, np.sum(c * pi_t))
                    cur_val = np.sum(c * pi_t)
                    pi = pi_t
            print('Transport cost overall', np.sum(c * pi * np.max(pi.shape)))  # pi.shape thing should yield same oom as inertia
            pi_list.append(pi)


    input_length = net.get_incoming_weights(layers1[0], numpy=True).shape[1]
    kernels_forward = [np.identity(input_length)]
    kernels_backward = [np.identity(input_length)]
    for l in range(L):
        layer_name = layers1[l]
        acti_for_len = net.get_activations(layer_name, t_dat, numpy=True).T
        len_act, _ = acti_for_len.shape
        if l == L-1:
            kernels_forward.append(np.identity(len_act))
            kernels_backward.append(np.identity(len_act))
        else:
            pi = pi_list[l]
            print(pi.shape)
            w_mu = np.sum(pi, axis=1)
            w_quant = np.sum(pi, axis=0) + tol

            k_for = np.transpose(pi / w_mu[:, None])  # kernel forward resulting from pi, so pi = mu times k_for
            k_back = np.divide(pi, w_quant[None, :])  # kernel backward resulting from pi, so pi = nu times k_back
            kernels_forward.append(k_for.copy())
            kernels_backward.append(k_back.copy())

    weight_list = []
    for l in range(L):
        k_back_incoming = kernels_backward[l]
        k_for_outgoing = kernels_forward[l+1]
        w_a = net.get_incoming_weights(layers1[l], numpy=True)
        w_b = k_for_outgoing @ w_a @ k_back_incoming
        weight_list.append(w_b.copy())
    out_net = Deep_MLP(hidden_size_1=smaller_sizes[0], hidden_size_2=smaller_sizes[1], hidden_size_3=smaller_sizes[2], assign_w=True, w_mats=weight_list, which_act=which_act)
    return out_net

def structured_pruning_ff(net, smaller_sizes, post_proc=False, use_test=False, test_data=[], which_act=0, out_going_weights=1, weights=None):
    layers1 = net.get_layer_names_with_weights()
    L = len(layers1)
    pre_clusters = []
    for l in range(L):
        if use_test:
            w_a = net.get_activations(layers1[l], test_data, numpy=True).T
        elif out_going_weights == 0:
            w_a = net.get_incoming_weights(layers1[l], numpy=True)
        else:
            if l < L-1:
                w_a = net.get_incoming_weights(layers1[l+1], numpy=True).T

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
            if use_test:
                norm_h = np.std(w_a, axis=1)
            elif out_going_weights == 0:
                norm_h = np.linalg.norm(w_a[:, inds_in], axis=1)
            else:
                norm_h = np.linalg.norm(w_a, axis=1)
            if weights is not None:
                norm_h *= weights[l]
            sort_inds = np.argsort(norm_h)[::-1]
            inds_out = sort_inds[:smaller_sizes[l]]
        pre_clusters.append(inds_out.copy())

    if post_proc:
        input_length = net.get_incoming_weights(layers1[0], numpy=True).shape[1]
        kernels_forward = [np.identity(input_length)]
        kernels_backward = [np.identity(input_length)]

        if which_act==0:
            act_h = torch.nn.ReLU()
        elif which_act==1:
            act_h = torch.nn.LeakyReLU()
        elif which_act==2:
            act_h = torch.nn.GELU()

        for l in range(L-1):
            w_out = net.get_incoming_weights(layers1[l+1], numpy=True).copy()

            # w_in = net.get_incoming_weights(layers1[l], numpy=True).copy()
            layer_name = layers1[l]
            activations = net.get_activations(layer_name, test_data).T  # doesn't work as well here
            activations = act_h(activations)
            activations = activations.detach().cpu().numpy()

            state = activations  # could also use w_in, np.transpose(w_out) or activations, but w_in seems best here

            x_full = state
            x_centers = state[pre_clusters[l+1], :]
            c = ot.dist(x_full, x_centers,
                        metric='sqeuclidean')
            w_full = np.ones(len(state))/len(state)
            w_centers = np.ones(len(x_centers))/len(x_centers)
            pi_t = ot.emd(w_full, w_centers, c)
            w_mu = np.sum(pi_t, axis=1)
            w_quant = np.sum(pi_t, axis=0)
            k_for = np.transpose(pi_t / w_mu[:, None])  # kernel forward resulting from pi, so pi = mu times k_for
            k_back = pi_t / w_quant[None, :]  # kernel backward resulting from pi, so pi = nu times k_back
            kernels_forward.append(k_for.copy())
            kernels_backward.append(k_back.copy())
        kernels_forward.append(np.identity(len(w_out)))
        kernels_backward.append(np.identity(len(w_out)))
        weight_list = []
        for l in range(L):
            k_back_incoming = kernels_backward[l]
            k_for_outgoing = kernels_forward[l + 1]
            w_a = net.get_incoming_weights(layers1[l], numpy=True)
            w_b = k_for_outgoing @ w_a @ k_back_incoming
            weight_list.append(w_b.copy())
        out_net = Deep_MLP(hidden_size_1=smaller_sizes[0], hidden_size_2=smaller_sizes[1],
                           hidden_size_3=smaller_sizes[2], assign_w=True, w_mats=weight_list, which_act=which_act)
    else:
        weight_list = []
        for l in range(L):
            w_a = net.get_incoming_weights(layers1[l], numpy=True)
            w_b = w_a[:, pre_clusters[l]][pre_clusters[l+1], :]
            weight_list.append(w_b.copy())
        out_net = Deep_MLP(hidden_size_1=smaller_sizes[0], hidden_size_2=smaller_sizes[1], hidden_size_3=smaller_sizes[2], assign_w=True, w_mats=weight_list, which_act=which_act)
    return out_net