import os
import sys
from matplotlib import pyplot as plt

sys.path.append("../")
from src.FusionModel.generalized_pruning.pruning_mlp import *
from src.FusionModel.fusion_methods.naive_fusion import NaiveFusion
from src import data_loader
from src.MLP import Deep_MLP
from src.FusionModel.fusion_methods import PartialFusion
from src.FusionModel import FusionModel


np.random.seed(0)
torch.random.manual_seed(0)

SPECIALIST = 1
seeds = [0, 1, 2, 3, 4]
alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
act = 2  # 0 is Relu, 1 is LeakyRelu, 2 is Gelu
lambdas = [0, 1]
USE_IMP = True
which_algo = 'stoch_very_high'

features = ''  # if empty then same as 'act', otherwise 'both' or 'weight_based'

os.makedirs('results_ens_compression', exist_ok=True)
save_path = 'results_ens_compression/' + str(seeds) +'_' + str(alphas) + '_' + str(act) + '_' + str(USE_IMP) + '_' + str(which_algo) + features
print(save_path)

list_prune = []
list_post_proc = []
list_free_act = []
list_partial_fusion = []
list_naive = []

for i_seed, seed in enumerate(seeds):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    train_loader, test_loader = data_loader.load_mnist()
    current_seed_prune = []
    current_seed_post = []
    current_seed_free_act = []
    current_seed_partial_fusion = []
    current_seed_naive = []

    test_data = []
    for batch in train_loader:
        inputs = batch[0]
        test_data.append(inputs)
    test_data = torch.cat(test_data, dim=0)[:1000]


    large_size = [100, 100, 100]
    ACT_SAVE = 1

    model_a = Deep_MLP(hidden_size_1=large_size[0], hidden_size_2=large_size[1], hidden_size_3=large_size[2], which_act=act)
    model_b = Deep_MLP(hidden_size_1=large_size[0], hidden_size_2=large_size[1],
                        hidden_size_3=large_size[2], which_act=act)
    if SPECIALIST:
        if ACT_SAVE:
            model_a.load_model('saved_compression/' + str(seed) + 'deepmlpmnist_general_' + str(large_size) + '_' + str(act) + '.checkpoint')
            model_b.load_model('saved_compression/' + str(seed) + 'deepmlpmnist_specific_' + str(large_size) + '_' + str(act) + '.checkpoint')
        else:
            model_a.load_model('saved_compression/' + str(seed) + 'deepmlpmnist_general_' + str(large_size) + '.checkpoint')
            model_b.load_model('saved_compression/' + str(seed) + 'deepmlpmnist_specific_' + str(large_size) + '.checkpoint')

    else:
        if ACT_SAVE:
            model_a.load_model('saved_compression/' + str(2*seed) + 'deepmlpmnist_' + str(large_size) +'_'+str(act) + '.checkpoint')
            model_b.load_model('saved_compression/' + str(2*seed+1) + 'deepmlpmnist_' + str(large_size) +'_'+str(act) + '.checkpoint')
        else:
            model_a.load_model('saved_compression/' + str(2*seed) + 'deepmlpmnist_' + str(large_size) + '.checkpoint')
            model_b.load_model('saved_compression/' + str(2*seed+1) + 'deepmlpmnist_' + str(large_size) + '.checkpoint')

    acc_a = model_a.test_model(test_loader)
    acc_b = model_b.test_model(test_loader)
    model_ens = FusionModel(model_a, model_b, PartialFusion(alphas=1), lambdas=lambdas)
    ens_acc = model_ens.test_model(test_loader)
    for a in alphas:
        mult = 1 + a
        small_size = [int(np.round(mult * l)) for l in large_size]
        if USE_IMP:
            importance = []
            for l in large_size:
                l_imp = np.ones(2*l)
                split_point = int(l)
                l_imp[:split_point] *= min(lambdas)
                l_imp[split_point:] *= max(lambdas)
                importance.append(l_imp)
        else:
            importance = None

        print('----------- Seed:' + str(seed) + ', Size: ' + str(small_size) + '---------------')
        naive_fusion = FusionModel(
                    model_a, model_b,
                    NaiveFusion(),
                    lambdas=lambdas
                )
        val_naive = naive_fusion.test_model(test_loader)
        model_raw_prune = structured_pruning_ff(model_ens, small_size, use_test=True, test_data=test_data, which_act=act, weights=importance)
        val_prune = model_raw_prune.test_model(test_loader)
        model_raw_prune_postproc = structured_pruning_ff(model_ens, small_size, post_proc=True, use_test=True, test_data=test_data, which_act=act) #, weights=importance
        val_post = model_raw_prune_postproc.test_model(test_loader)
        if features == '' or features == 'act':
            model_act = prune_net_clean(model_ens, small_size, test_data, which_act=act, importance=importance, use_reg=False, clustering_algo=which_algo)
        elif features == 'weight_based':
            model_act = prune_net_clean(model_ens, small_size, test_data, weight_based=True, which_act=act, importance=importance,
                                        use_reg=False, clustering_algo=which_algo)
        elif features == 'both':
            model_act = prune_net_clean(model_ens, small_size, test_data, which_act=act, importance=importance, use_reg=False, clustering_algo=which_algo, both=True)

        val_act = model_act.test_model(test_loader)

        model_partial_fusion = FusionModel(
                    model_a, model_b,
                    PartialFusion(alphas=a, combine_costs=True,
                    pgd=True),
                    lambdas=lambdas,
                )
        val_pf = model_partial_fusion.test_model(test_loader)
        current_seed_prune.append(val_prune)
        current_seed_post.append(val_post)
        current_seed_free_act.append(val_act)
        current_seed_partial_fusion.append(val_pf)
        current_seed_naive.append(val_naive)

    list_naive.append(current_seed_naive)
    list_prune.append(current_seed_prune)
    list_post_proc.append(current_seed_post)
    list_free_act.append(current_seed_free_act)
    list_partial_fusion.append(current_seed_partial_fusion)

np.save(save_path+'_prune', np.array(list_prune))
np.save(save_path+'_post', np.array(list_post_proc))
np.save(save_path+'_free', np.array(list_free_act))
np.save(save_path+'_partial_fusion', np.array(list_partial_fusion))

naivenp = np.mean(np.array(list_naive), axis=0)
print(naivenp)
print(np.std(np.array(list_naive), axis=0))
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