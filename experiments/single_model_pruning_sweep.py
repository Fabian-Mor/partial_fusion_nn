import sys
import copy
import torch
import matplotlib as mpl
from fontTools.misc.cython import returns

mpl.rcParams['figure.dpi'] = 300
import numpy as np
import torch
sys.path.append("../")

from src.CNN import VGG11
from src.FusionModel.fusion_methods.naive_fusion import NaiveFusion
from src import data_loader
from src.MLP import Deep_MLP, MLP
from src.FusionModel.fusion_model import FusionModel
from src.FusionModel.fusion_methods.partial_fusion import PartialFusion
from src.FusionModel.generalized_pruning import StochHierarchical
from src.FusionModel.generalized_pruning import WeightHierarchical
from src.FusionModel.generalized_pruning import StructuredPruning

CNN = True

if CNN:
    train_loader, test_loader = data_loader.load_cifar10()
    model_cls = VGG11
else:
    train_loader, test_loader = data_loader.load_mnist()
    model_cls = Deep_MLP

test_data = []
for batch in train_loader:
    inputs = batch[0]
    test_data.append(inputs)
test_data = torch.cat(test_data, dim=0)[:1000]
criterion = None #torch.nn.CrossEntropyLoss()
save = False

pruning_method =  'stoch' #'structured', 'ot paper', 'stoch'
act=2
retrain_epochs=100

mults = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

for i in range(5):
    if CNN:
        model_size = [64, 128, 256, 256, 512, 512, 512, 512]
        model = VGG11()
        model.load_state_dict(
            torch.load('saved_compression/' + str(i) + 'VGG11_' + str(model_size) + '_best.checkpoint'))
    else:
        model_size = [100, 100, 100]
        model = Deep_MLP(hidden_size_1=100, hidden_size_2=100, hidden_size_3=100, which_act=act)
        if save:
            model.train_model_best_ckpt(train_loader, test_loader, epochs=5)

            model.save_model(f'saved/model_{i}')
        else:
            model.load_model('saved_compression/' + str(i) + 'deepmlpmnist_' + str([100, 100, 100]) + '_' + str(
                        act) + '.checkpoint')

    test_a = model.test_model(test_loader, criterion=criterion)
    for mult in mults:
        if pruning_method == 'structured':
            pruned_model = StructuredPruning().prune_network(model, model_size, mult, model_cls, test_data,
                                                             which_act=act, post_proc=False)
        elif pruning_method == 'ot paper':
            pruned_model = StructuredPruning().prune_network(model, model_size, mult, model_cls, test_data,
                                                             which_act=act, post_proc=True)
        elif pruning_method == 'stoch':
            pruned_model = StochHierarchical(n_restarts=50000).prune_network(model, model_size, mult, model_cls, test_data, which_act=act)
        pruned_model.test_model(test_loader, criterion=criterion)
        if train_loader is not None and retrain_epochs > 0:
            _, acc = pruned_model.train_model_best_ckpt(train_loader, test_loader, epochs=retrain_epochs, verbose=False)
            print(acc)