"""
Ablation study: MLP capacity (uniform hidden width) for partial fusion.

Structure follows partial_fusion_sweep.py, but the outer axis is the hidden
size of the MLP (all three hidden layers tied to the same width). For each
capacity value, we train num_seeds pairs of Deep_MLPs on MNIST, run the
fusion sweep over (lambda, alpha), and report mean +- std across seeds.
"""
import sys
import os
import copy
import time
import torch
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np

sys.path.append("../")
sys.stdout.reconfigure(line_buffering=True)

from src import data_loader
from src.MLP import Deep_MLP
from src.FusionModel.fusion_model import FusionModel
from src.FusionModel.fusion_methods.naive_fusion import NaiveFusion
from src.FusionModel.fusion_methods.partial_fusion import PartialFusion


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
# Configuration
# =====================================================================

act = 2                                           # 0=ReLU, 1=LeakyReLU, 2=GELU
num_seeds = 5
hidden_sizes = [25, 50, 100, 200, 400]            # capacity axis
lambdas = [0, 0.5, 1]
alphas = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
train_epochs = 5
retrain_epochs = 10
save_dir = 'saved'

os.makedirs(save_dir, exist_ok=True)

train_loader, test_loader = data_loader.load_mnist()
retrain_data_loader = train_loader

criterion = None  # accuracy


def ckpt_path(which, i, h):
    return os.path.join(save_dir, f'mlp_capacity_{which}_h{h}_s{i}.checkpoint')


def get_or_train(i, h, which):
    """Load or train one Deep_MLP with uniform hidden size h, seed-determined by (i, which).

    Robust to corrupt checkpoints (e.g. from an interrupted previous save):
    if torch.load fails, the file is deleted and we retrain. Saves are atomic
    (write to .tmp + os.replace) so future interruptions cannot leave a
    partially-written checkpoint behind.
    """
    model = Deep_MLP(hidden_size_1=h, hidden_size_2=h, hidden_size_3=h, which_act=act)
    path = ckpt_path(which, i, h)
    if os.path.exists(path):
        try:
            model.load_model(path)
            return model
        except (OSError, RuntimeError, EOFError, RuntimeError) as e:
            print(f"    [warn] corrupt checkpoint at {path}: {type(e).__name__}: {e}; retraining")
            try:
                os.remove(path)
            except OSError:
                pass
            # Re-instantiate the model so partial state from a failed load
            # cannot contaminate the fresh training run.
            model = Deep_MLP(hidden_size_1=h, hidden_size_2=h, hidden_size_3=h, which_act=act)
    seed = 2 * i if which == 'a' else 2 * i + 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    best, best_acc = model.train_model_best_ckpt(
        train_loader, test_loader, epochs=train_epochs, verbose=False
    )
    # Atomic save so an interruption mid-write can never produce a corrupt
    # checkpoint that breaks future runs.
    tmp_path = path + '.tmp'
    best.save_model(tmp_path)
    os.replace(tmp_path, path)
    print(f"    trained {which} (seed={seed}) -> {best_acc:.2f}%")
    return best


def _fmt_row(mean_arr, std_arr):
    """Format a per-lambda row as 'mm.mm±s.ss' columns."""
    return "  ".join(f"{m:6.2f}±{s:4.2f}" for m, s in zip(mean_arr, std_arr))


def _print_summary(h, r, num_seeds, lambdas):
    """Aligned per-capacity summary table."""
    col_w = 12
    lam_header = "  ".join(f"λ={l:<5.2f}".ljust(col_w) for l in lambdas)
    print(f"\n  hidden = {h}  ({num_seeds} seeds)")
    print(f"    {'metric':<18s}  {lam_header}")
    print(f"    {'-'*18}  {'-'*len(lam_header)}")
    print(f"    {'Model A':<18s}  {_fmt_row(r['a_mean'], r['a_std'])}")
    print(f"    {'Model B':<18s}  {_fmt_row(r['b_mean'], r['b_std'])}")
    print(f"    {'Naive fusion':<18s}  {_fmt_row(r['naive_mean'], r['naive_std'])}")
    for alpha in sorted(r['fuse_mean'].keys()):
        label = f"fuse α={alpha}"
        print(f"    {label:<18s}  {_fmt_row(r['fuse_mean'][alpha], r['fuse_std'][alpha])}")
    if r['retrain_mean']:
        for alpha in sorted(r['retrain_mean'].keys()):
            label = f"retrain α={alpha}"
            print(f"    {label:<18s}  {_fmt_row(r['retrain_mean'][alpha], r['retrain_std'][alpha])}")


# =====================================================================
# Main loop: outer = capacity, inner = seeds
# =====================================================================

capacity_results = {}

for h in hidden_sizes:
    print(f"\n{'='*60}\n  Capacity: hidden = {h}\n{'='*60}")

    acc_fuses = []
    acc_naives = []
    acc_as = []
    acc_bs = []
    acc_trains = []

    for i in range(num_seeds):
        seed_t0 = time.time()
        print(f"\n  seed {i+1}/{num_seeds}")
        model_a = get_or_train(i, h, 'a')
        model_b = get_or_train(i, h, 'b')

        test_a = model_a.test_model(test_loader, criterion=criterion, verbose=False)
        test_b = model_b.test_model(test_loader, criterion=criterion, verbose=False)
        print(f"    A={test_a:.2f}%  B={test_b:.2f}%", end="", flush=True)
        acc_as.append([test_a] * len(lambdas))
        acc_bs.append([test_b] * len(lambdas))

        acc_fuse = {}
        acc_train = {}
        acc_naive = []
        for l in lambdas:
            naive_model = FusionModel(model_a, model_b, NaiveFusion(), lambdas=[1 - l, l])
            naive_acc = naive_model.test_model(test_loader, criterion=criterion, verbose=False)
            acc_naive.append(naive_acc)

            for alpha in alphas:
                fused_model = FusionModel(
                    model_a, model_b,
                    PartialFusion(alphas=alpha, combine_costs=True, pgd=True),
                    lambdas=[1 - l, l],
                )
                accuracy = fused_model.test_model(test_loader, verbose=False, criterion=criterion)
                acc_fuse.setdefault(alpha, []).append(accuracy)

                if retrain_data_loader is not None and retrain_epochs > 0:
                    freeze_zero_blocks(fused_model)
                    _, acc = fused_model.train_model_best_ckpt(
                        retrain_data_loader, test_loader,
                        epochs=retrain_epochs, verbose=False
                    )
                    acc_train.setdefault(alpha, []).append(acc)

        acc_fuses.append(acc_fuse)
        acc_naives.append(acc_naive)
        acc_trains.append(acc_train)
        print(f"  ({time.time() - seed_t0:.0f}s)", flush=True)

    # ----- aggregate over seeds -----
    acc_naives_arr = np.array(acc_naives)
    acc_as_arr = np.array(acc_as)
    acc_bs_arr = np.array(acc_bs)

    fuse_mean = {}
    fuse_std = {}
    retrain_mean = {}
    retrain_std = {}
    for alpha in sorted(acc_fuses[0].keys()):
        fuse_arr = np.array([acc_fuses[r][alpha] for r in range(num_seeds)])
        fuse_mean[alpha] = np.mean(fuse_arr, axis=0)
        fuse_std[alpha] = np.std(fuse_arr, axis=0)
        if retrain_epochs > 0 and acc_trains[0].get(alpha):
            retrain_arr = np.array([acc_trains[r][alpha] for r in range(num_seeds)])
            retrain_mean[alpha] = np.mean(retrain_arr, axis=0)
            retrain_std[alpha] = np.std(retrain_arr, axis=0)

    capacity_results[h] = {
        'a_mean': np.mean(acc_as_arr, axis=0), 'a_std': np.std(acc_as_arr, axis=0),
        'b_mean': np.mean(acc_bs_arr, axis=0), 'b_std': np.std(acc_bs_arr, axis=0),
        'naive_mean': np.mean(acc_naives_arr, axis=0),
        'naive_std': np.std(acc_naives_arr, axis=0),
        'fuse_mean': fuse_mean,
        'fuse_std': fuse_std,
        'retrain_mean': retrain_mean,
        'retrain_std': retrain_std,
    }

    _print_summary(h, capacity_results[h], num_seeds, lambdas)


# =====================================================================
# Final cross-capacity summary
# =====================================================================

print("\n" + "=" * 72)
print("CAPACITY ABLATION SUMMARY")
print("=" * 72)
for h in hidden_sizes:
    if h in capacity_results:
        _print_summary(h, capacity_results[h], num_seeds, lambdas)
