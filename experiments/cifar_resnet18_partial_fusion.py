"""
Experiment script for partial fusion of two ResNet18 models on CIFAR10.

Usage is identical to the existing VGG11 experiment — the only change is
swapping VGG11() for ResNet18() and passing a DataLoader as the `data`
argument so that BatchNorm layers get recalibrated after fusion.

Two modes:
  - save=True:  trains two ResNet18 models from scratch, saves checkpoints
  - save=False: loads pretrained checkpoints and runs fusion experiments
"""

import sys
import copy
import torch
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np

sys.path.append("../")

from src.CNN import VGG11
from src.CNN.ResNet import ResNet18
from src.FusionModel.fusion_methods.naive_fusion import NaiveFusion
from src import data_loader
from src.FusionModel.fusion_model import FusionModel
from src.FusionModel.fusion_methods.partial_fusion import PartialFusion
from src.FusionModel.generalized_pruning import StochHierarchical


# =====================================================================
# Configuration
# =====================================================================

save = False           # True: train models from scratch; False: load checkpoints
feature_base = 'pcd'  # 'pcd', 'weight', 'activation'
retrain_epochs = 0
num_seeds = 5         # number of model pairs to average over

lambdas = [0.5] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
alphas = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
model_name = 'resnet18'

# =====================================================================
# Data
# =====================================================================

train_loader, test_loader = data_loader.load_cifar10(batch_size=128)

# Calibration data for BatchNorm recalibration after fusion.
# We reuse the train_loader directly — the fusion model will do a forward
# pass in train mode to collect BN running statistics.
# For activation-based features, we also need a tensor batch:
calib_data = []
for batch in train_loader:
    calib_data.append(batch[0])
    if len(calib_data) * batch[0].shape[0] >= 1000:
        break
calib_data = torch.cat(calib_data, dim=0)[:1000]

criterion = None  # None = use accuracy (default in BaseModel.test_model)


# =====================================================================
# Helper: freeze zero blocks for retraining after fusion
# =====================================================================

def freeze_zero_blocks(model):
    for param in model.parameters():
        if param.requires_grad:
            mask = (param.data != 0).type_as(param.data)
            def create_hook(m):
                def hook(grad):
                    return grad * m
                return hook
            param.register_hook(create_hook(mask))


# =====================================================================
# Main experiment loop
# =====================================================================

acc_fuses = []
acc_naives = []
acc_as = []
acc_bs = []
acc_trains = []

import os

for i in range(num_seeds):
    print(f"\n{'='*60}")
    print(f"  Seed pair {i}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Create / load two ResNet18 models
    # ------------------------------------------------------------------
    model_a = ResNet18(num_classes=10)
    model_b = ResNet18(num_classes=10)

    ckpt_a = f'saved/{model_name}_a_{i}.checkpoint'
    ckpt_b = f'saved/{model_name}_b_{i}.checkpoint'

    def _train_one(model, name, ckpt_path):
        # Standard ResNet18 / CIFAR-10 recipe:
        # SGD with lr=0.1, momentum=0.9, weight_decay=5e-4, cosine annealing, 200 epochs.
        print(f"Training model {name} (no checkpoint at {ckpt_path})...")
        model.optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model.optimizer, T_max=200
        )
        best, best_acc = model.train_model_best_ckpt(
            train_loader, test_loader, epochs=200, verbose=True
        )
        best.save_model(ckpt_path)
        print(f"Model {name} best accuracy: {best_acc:.2f}%")
        return best

    if save or not os.path.exists(ckpt_a):
        model_a = _train_one(model_a, 'A', ckpt_a)
    else:
        model_a.load_model(ckpt_a)

    if save or not os.path.exists(ckpt_b):
        model_b = _train_one(model_b, 'B', ckpt_b)
    else:
        model_b.load_model(ckpt_b)
    print(model_a)
    # ------------------------------------------------------------------
    # 2. Evaluate individual models
    # ------------------------------------------------------------------
    test_a = model_a.test_model(test_loader, criterion=criterion)
    test_b = model_b.test_model(test_loader, criterion=criterion)
    print(f"Model A accuracy: {test_a:.2f}%")
    print(f"Model B accuracy: {test_b:.2f}%")

    # Sanity check: fuse model A with itself at lambda=0 — should reproduce model A
    sanity = FusionModel(model_a, model_a, NaiveFusion(), lambdas=[1.0, 0.0], bn_data=train_loader)
    sanity_acc = sanity.test_model(test_loader, criterion=criterion, verbose=False)
    print(f"Sanity check (A fused with A, lambda=0): {sanity_acc:.2f}% (should be ~{test_a:.2f}%)")

    acc_a = [test_a] * len(lambdas)
    acc_b = [test_b] * len(lambdas)
    acc_as.append(acc_a)
    acc_bs.append(acc_b)

    # ------------------------------------------------------------------
    # 3. Fusion experiments
    # ------------------------------------------------------------------
    acc_fuse = {}
    acc_train = {}
    acc_naive = []

    for l in lambdas:
        print(f"\n  lambda = {l}")

        # Naive fusion (no alignment) — pass bn_data for BN recalibration
        naive_model = FusionModel(model_a, model_b, NaiveFusion(), lambdas=[1 - l, l], bn_data=train_loader)
        acc_naive.append(naive_model.test_model(test_loader, criterion=criterion))

        for alpha in alphas:
            print(f"    alpha = {alpha}", end="")

            # data: tensor for activation features (None for weight-based methods)
            # bn_data: DataLoader for BN recalibration (always needed for ResNet)

            if feature_base == 'pcd':
                #alpha_l = [alpha] * 28
                #alpha_l[0] = 1.0
                method = PartialFusion(alphas=alpha, combine_costs=True, pgd=True,
                                       tied_permutations=True, warmstart=True)
                fused_model = FusionModel(
                    model_a, model_b, method,
                    lambdas=[1 - l, l],
                    #running_stats_init='cosine',
                    bn_data=train_loader,  # BN recalibration
                )
            elif feature_base == 'weight':
                method = PartialFusion(alphas=alpha)
                fused_model = FusionModel(
                    model_a, model_b, method,
                    lambdas=[1 - l, l],
                    bn_data=train_loader,
                )
            elif feature_base == 'activation':
                method = PartialFusion(alphas=alpha)
                fused_model = FusionModel(
                    model_a, model_b, method,
                    lambdas=[1 - l, l],
                    data=calib_data,         # tensor for activation features
                    bn_data=train_loader,    # DataLoader for BN recalibration
                )
            else:
                # Clustering-based generalized pruning
                method = StochHierarchical(None, alphas=alpha)
                fused_model = FusionModel(
                    model_a, model_b, method,
                    lambdas=[1 - l, l],
                    data=calib_data,
                    bn_data=train_loader,
                )

            accuracy = fused_model.test_model(test_loader, verbose=False, criterion=criterion)
            acc_fuse.setdefault(alpha, []).append(accuracy)

            total_w = fused_model.get_total_weights()
            nonzero_w = fused_model.non_zero_weights
            print(f"  -> acc={accuracy:.2f}%  params={total_w}  nonzero={nonzero_w}")

            # Optional retraining
            if retrain_epochs > 0:
                freeze_zero_blocks(fused_model)
                _, acc = fused_model.train_model_best_ckpt(
                    train_loader, test_loader, epochs=retrain_epochs, verbose=False
                )
                print(f"       after retraining: {acc:.2f}%")
                acc_train.setdefault(alpha, []).append(acc)

    acc_fuses.append(acc_fuse)
    acc_naives.append(acc_naive)
    acc_trains.append(acc_train)


# =====================================================================
# Aggregate and print results
# =====================================================================

acc_naives_arr = np.array(acc_naives)
acc_as_arr = np.array(acc_as)
acc_bs_arr = np.array(acc_bs)

acc_naives_avg = np.mean(acc_naives_arr, axis=0)
acc_naives_std = np.std(acc_naives_arr, axis=0)
acc_as_avg = np.mean(acc_as_arr, axis=0)
acc_as_std = np.std(acc_as_arr, axis=0)
acc_bs_avg = np.mean(acc_bs_arr, axis=0)
acc_bs_std = np.std(acc_bs_arr, axis=0)

print("\n" + "="*60)
print(f"RESULTS  (averaged over {num_seeds} seeds)")
print("="*60)
print(f"Model A:       mean={acc_as_avg}  std={acc_as_std}")
print(f"Model B:       mean={acc_bs_avg}  std={acc_bs_std}")
print(f"Naive fusion:  mean={acc_naives_avg}  std={acc_naives_std}")

sink_weights_list = sorted(acc_fuses[0].keys())
results = {}
results_std = {}
result_retrain = {}
result_retrain_std = {}
for alpha in sink_weights_list:
    acc_fuse_values = []
    acc_retrain_values = []
    for run in range(len(acc_fuses)):
        acc_fuse_values.append(acc_fuses[run][alpha])
        if acc_trains[run].get(alpha):
            acc_retrain_values.append(acc_trains[run][alpha])
    acc_fuse_arr = np.array(acc_fuse_values)
    results[alpha] = np.mean(acc_fuse_arr, axis=0)
    results_std[alpha] = np.std(acc_fuse_arr, axis=0)
    if acc_retrain_values:
        acc_retrain_arr = np.array(acc_retrain_values)
        result_retrain[alpha] = np.mean(acc_retrain_arr, axis=0)
        result_retrain_std[alpha] = np.std(acc_retrain_arr, axis=0)
    print(f"  alpha={alpha}: mean={results[alpha]}  std={results_std[alpha]}")

print("\nFusion results (mean):", results)
print("Fusion results (std):", results_std)
print("Retrain results (mean):", result_retrain)
print("Retrain results (std):", result_retrain_std)
