import sys
import copy
import torch
import matplotlib as mpl

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


def freeze_zero_blocks(model):
    for param in model.parameters():
        if param.requires_grad:
            mask = (param.data != 0).type_as(param.data)

            def create_hook(m):
                def hook(grad):
                    return grad * m

                return hook

            param.register_hook(create_hook(mask))


def freeze_zero_blocks_except_last(model):
    param_names = [name for name, _ in model.named_parameters()]

    if not param_names:
        return

    last_layer_prefix = param_names[-1].rsplit('.', 1)[0] if '.' in param_names[-1] else param_names[-1]

    for name, param in model.named_parameters():
        if name.startswith(last_layer_prefix):
            continue

        if param.requires_grad:
            mask = (param.data != 0).type_as(param.data)

            def create_hook(m):
                def hook(grad):
                    return grad * m

                return hook

            param.register_hook(create_hook(mask))


def freeze_zero_blocks_and_freeze_last(model):
    param_names = [name for name, _ in model.named_parameters()]

    if not param_names:
        return

    last_layer_prefix = param_names[-1].rsplit('.', 1)[0] if '.' in param_names[-1] else param_names[-1]

    for name, param in model.named_parameters():
        if name.startswith(last_layer_prefix):
            param.requires_grad = False
        else:
            if param.requires_grad:
                mask = (param.data != 0).type_as(param.data)

                def create_hook(m):
                    def hook(grad):
                        return grad * m

                    return hook

                param.register_hook(create_hook(mask))

CNN = False

if CNN:
    train_loader, test_loader = data_loader.load_cifar10()
else:
    train_loader, test_loader = data_loader.load_mnist()

test_data = []
for batch in train_loader:
    inputs = batch[0]
    test_data.append(inputs)
test_data = torch.cat(test_data, dim=0)[:10000]
criterion = None #torch.nn.CrossEntropyLoss()
save = False

feature_base =  'weight_hierarch' # 'weight', 'activation', 'pcd', 'prune', 'weight_hierarch'
SPECIALIST = True
act=2

if SPECIALIST:
    retrain_data_loader = data_loader.load_mnist_subset(percentage=0.01)[0]
else:
    retrain_data_loader = train_loader
retrain_epochs = 0 #10

lambdas = [0, 0.5, 1] #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
alphas = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]

acc_fuses = []
acc_naives = []
acc_as = []
acc_bs = []
acc_trains = []

for i in range(5):

    if CNN:
        model_size = [64, 128, 256, 256, 512, 512, 512, 512]
        model_a = VGG11()
        model_b = VGG11()
        model_a.load_state_dict(
            torch.load('saved_compression/' + str(2 * i) + 'VGG11_' + str(model_size) + '_best.checkpoint'))
        model_b.load_state_dict(
            torch.load('saved_compression/' + str(2 * i + 1) + 'VGG11_' + str(model_size) + '_best.checkpoint'))
    else:
        model_a = Deep_MLP(hidden_size_1=100, hidden_size_2=100, hidden_size_3=100, which_act=act) #Deep_MLP()
        model_b = Deep_MLP(hidden_size_1=100, hidden_size_2=100, hidden_size_3=100, which_act=act) #Deep_MLP()
        if save:
            model_a.train_model_best_ckpt(train_loader, test_loader, epochs=5)
            model_b.train_model_best_ckpt(train_loader, test_loader, epochs=5)

            model_a.save_model(f'saved/model_a_{i}')
            model_b.save_model(f'saved/model_b_{i}')
        else:
            if SPECIALIST:
                model_b.load_model('saved_compression/'+str(i)+'deepmlpmnist_general_'+str([100, 100, 100])+'_'+str(act)+'.checkpoint') #model_a.load_model(f'saved/model_a_{i}')
                model_a.load_model('saved_compression/'+str(i)+'deepmlpmnist_specific_'+str([100, 100, 100])+'_'+str(act)+'.checkpoint') #model_b.load_model(f'saved/model_b_{i}')
            else:
                model_a.load_model('saved_compression/' + str(2*i) + 'deepmlpmnist_' + str([100, 100, 100]) + '_' + str(act) + '.checkpoint')
                model_b.load_model('saved_compression/' + str(2*i+1) + 'deepmlpmnist_' + str([100, 100, 100]) + '_' + str(act) + '.checkpoint')

    test_a = model_a.test_model(test_loader, criterion=criterion)
    test_b = model_b.test_model(test_loader, criterion=criterion)
    # lambda values and model accuracies
    acc_a = [test_a] * len(lambdas)
    acc_b = [test_b] * len(lambdas)
    acc_as.append(acc_a)
    acc_bs.append(acc_b)

    # Fused model accuracies
    acc_fuse = {}
    acc_train = {}
    acc_naive = []
    for l in lambdas:
        naive_model = FusionModel(model_a, model_b, NaiveFusion(), lambdas=[1 - l, l])
        acc_naive.append(naive_model.test_model(test_loader, criterion=criterion))
        for alpha in alphas:
            if feature_base == 'pcd':
                fused_model = FusionModel(
                    model_a, model_b,
                    PartialFusion(alphas=alpha, combine_costs=True, pgd=True),
                    lambdas=[1 - l, l],
                )
            elif feature_base == 'weight':
                fused_model = FusionModel(
                    model_a, model_b,
                    PartialFusion(alphas=alpha),
                    lambdas=[1 - l, l],
                )
            elif feature_base == 'activation':
                fused_model = FusionModel(
                    model_a, model_b,
                    PartialFusion(alphas=alpha),
                    lambdas=[1 - l, l],
                    data=test_data,
                )
            elif feature_base == 'prune':
                fused_model = FusionModel(
                    model_a, model_b,
                    StructuredPruning(alphas=alpha),
                    lambdas=[1 - l, l],
                )
            elif feature_base == 'weight_hierarch':
                fused_model = FusionModel(
                    model_a, model_b,
                    WeightHierarchical(alphas=alpha),
                    lambdas=[1 - l, l],
                    data=test_data
                )
            else:
                fused_model = FusionModel(
                    model_a, model_b,
                    StochHierarchical(None, alphas=alpha),
                    lambdas=[1 - l, l],
                    data=test_data
                )
            accuracy = fused_model.test_model(test_loader, verbose=False, criterion=criterion)
            acc_fuse.setdefault(alpha, []).append(accuracy)
            print(fused_model.get_total_weights())
            print(fused_model.non_zero_weights)
            print(l, alpha, accuracy)
            if retrain_data_loader is not None and retrain_epochs > 0:
                freeze_zero_blocks(fused_model)
                _, acc = fused_model.train_model_best_ckpt(retrain_data_loader, test_loader, epochs=retrain_epochs, verbose=False)
                print('After retraining: ', acc)
                acc_train.setdefault(alpha, []).append(acc)

    acc_fuses.append(acc_fuse)
    acc_naives.append(acc_naive)
    acc_as.append(acc_a)
    acc_bs.append(acc_b)
    acc_trains.append(acc_train)

acc_naives = np.mean(np.array(acc_naives), axis=0)
acc_as = np.mean(np.array(acc_as), axis=0)
acc_bs = np.mean(np.array(acc_bs), axis=0)
print(acc_trains)
print(acc_naives)
print(acc_as)
print(acc_bs)

sink_weights_list = sorted(acc_fuses[0].keys())
results = {}
result_retrain = {}
for i, alpha in enumerate(sink_weights_list):
    acc_fuse_values = []
    acc_retrain_values = []
    for run in range(len(acc_fuses)):
        acc_fuse_values.append(acc_fuses[run][alpha])
        acc_retrain_values.append(acc_trains[run][alpha])
    acc_fuse_values = np.mean(np.array(acc_fuse_values), axis=0)
    acc_retrain_values = np.mean(np.array(acc_retrain_values), axis=0)
    print(alpha, acc_fuse_values)
    results[alpha] = acc_fuse_values
    result_retrain[alpha] = acc_retrain_values
print(results)
print(result_retrain)