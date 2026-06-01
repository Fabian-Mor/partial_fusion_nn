import os
import sys
import torch
from matplotlib import pyplot as plt

sys.path.append("../")
from src.FusionModel.generalized_pruning.pruning_cnn import *
from src import data_loader
from src.CNN import VGG11
from src.FusionModel.fusion_methods import PartialFusion
from src.FusionModel import FusionModel


np.random.seed(0)
torch.random.manual_seed(0)

SPECIALIST = 0
seeds = [0]  # only 2 VGG11 baselines (seeds 0,1) shipped → 1 pair (uses indices 2*seed, 2*seed+1)
alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
lambdas = [0.2, 0.8]
USE_IMP = True
which_algo = 'stoch_very_high'  # 'stoch', 'stoch_high', 'stoch_very_high', 'redi', 'stoch_redi', 'stoch_redi_high' 'annealing', 'kmeans' (other strings: greedy)

features = ''  # if empty then same as 'act', otherwise 'both' or 'weight_based'

os.makedirs('results', exist_ok=True)
save_path = 'results/ens_cnn' + str(seeds) +'_' + str(alphas) + '_' + str(0) + '_' + str(USE_IMP) + '_' + str(which_algo) + features
model_size = [64, 128, 256, 256, 512, 512, 512, 512]
print(save_path)

list_prune = []
list_post_proc = []
list_free_act = []
list_partial_fusion = []

for i_seed, seed in enumerate(seeds):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    train_loader, test_loader = data_loader.load_cifar10(batch_size=128)
    current_seed_prune = []
    current_seed_post = []
    current_seed_free_act = []
    current_seed_partial_fusion = []

    test_data = []
    for batch in train_loader:
        inputs = batch[0]
        test_data.append(inputs)
    test_data = torch.cat(test_data, dim=0)[:1000]

    model_a = VGG11()
    model_b = VGG11()
    model_a.load_state_dict(
        torch.load('saved/' + str(2*seed) + 'VGG11_' + str(model_size) + '_best.checkpoint'))
    model_b.load_state_dict(
        torch.load('saved/' + str(2*seed+1) + 'VGG11_' + str(model_size) + '_best.checkpoint'))
    model_a.test_model(test_loader)
    model_b.test_model(test_loader)
    model_ens = FusionModel(model_a, model_b, PartialFusion(alphas=1), lambdas=lambdas)
    model_ens.test_model(test_loader)
    for a in alphas:
        mults = [1 + a, 1 + a, 1 + a, 1 + a, 1 + a, 1 + a, 1 + a, 1 + a]
        small_size = [int(np.round(mult * l)) for l, mult in zip(model_size, mults)]
        if USE_IMP:
            importance = []
            for l in model_size:
                l_imp = np.ones(2*l)
                split_point = int(l)
                l_imp[:split_point] *= min(lambdas)
                l_imp[split_point:] *= max(lambdas)
                importance.append(l_imp)
        else:
            importance = None

        print('----------- Seed:' + str(seed) + ', Size: ' + str(small_size) + '---------------')
        model_raw_prune = prune_simple(model_ens, small_size, importance_scaling=importance, importance_type='incoming', fusion_network=True)
        val_prune = model_raw_prune.test_model(test_loader)
        model_act = prune_clustering(model_ens, small_size, test_data, clustering_algo=which_algo, importance_scaling=importance, fusion_network=True)

        val_act = model_act.test_model(test_loader)

        model_partial_fusion = FusionModel(
                model_a, model_b,
                PartialFusion(alphas=[a, a, a, a, a, a, a, a], combine_costs=True),
                lambdas=lambdas,
                pgd=True
                )
        val_pf = model_partial_fusion.test_model(test_loader)

        current_seed_prune.append(val_prune)
        current_seed_free_act.append(val_act)
        current_seed_partial_fusion.append(val_pf)

    list_prune.append(current_seed_prune)
    list_post_proc.append(current_seed_post)
    list_free_act.append(current_seed_free_act)
    list_partial_fusion.append(current_seed_partial_fusion)


np.save(save_path+'_prune', np.array(list_prune))
np.save(save_path+'_post', np.array(list_post_proc))
np.save(save_path+'_free', np.array(list_free_act))
np.save(save_path+'_partial_fusion', np.array(list_partial_fusion))

prunenp = np.mean(np.array(list_prune), axis=0)
postnp = np.mean(np.array(list_post_proc), axis=0)
actnp = np.mean(np.array(list_free_act), axis=0)
partial_fusion = np.mean(np.array(list_partial_fusion), axis=0)
print(prunenp)
print(postnp)
print(actnp)
print(partial_fusion)
print(np.std(np.array(list_prune), axis=0))
print(np.std(np.array(list_post_proc), axis=0))
print(np.std(np.array(list_free_act), axis=0))
print(np.std(np.array(list_partial_fusion), axis=0))