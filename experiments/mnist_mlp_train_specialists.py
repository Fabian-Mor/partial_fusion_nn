import sys
sys.path.append("../")

import torch
from src.FusionModel.generalized_pruning.pruning_mlp import *
from src import data_loader
from src.MLP import Deep_MLP

JUST_TEST = 0
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(seeds)
if not JUST_TEST:
    for seed in seeds:
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        general_train_loader, general_test_loader, specific_train_loader, specific_test_loader \
            = data_loader.split_mnist_by_digit(specific_digit=4, split_ratio=0.9)
        N_EPOCHS = 50
        SIZES = [[100, 100, 100]] # [[256, 128, 64], [128, 64, 32], [64, 32, 16], [32, 16, 10], [16, 16, 10], [10, 10, 10]]
        for size_h in SIZES:
            for act in [0, 1, 2]:  # 0 is Relu, 1 is LeakyRelu, 2 is Gelu
                model_a = Deep_MLP(hidden_size_1=size_h[0], hidden_size_2=size_h[1], hidden_size_3=size_h[2], which_act=act)
                model_a.train_model(general_train_loader, epochs=N_EPOCHS)
                model_a.save_model('saved/'+str(seed)+'deepmlpmnist_general_'+str(size_h)+'_'+str(act)+'.checkpoint')
                model_b = Deep_MLP(hidden_size_1=size_h[0], hidden_size_2=size_h[1], hidden_size_3=size_h[2],
                                   which_act=act)
                model_b.train_model(specific_train_loader, epochs=N_EPOCHS)
                model_b.save_model(
                    'saved/' + str(seed) + 'deepmlpmnist_specific_' + str(size_h) + '_' + str(act) + '.checkpoint')