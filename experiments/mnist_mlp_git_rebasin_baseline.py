"""
Sanity check for GitRebasin against PartialFusion.

Loads (or trains) two iid Deep_MLPs of uniform width h on MNIST, then:
  1. Verifies GitRebasin(λ=[1, 0])  ≈ model A   (single-model identity)
  2. Verifies GitRebasin(λ=[0, 1])  ≈ model B
  3. Reports test accuracy at λ=[0.5, 0.5] for:
       - NaiveFusion (no alignment)
       - PartialFusion across a few alphas
       - GitRebasin (no alpha)
     and prints them side-by-side. GitRebasin should sit near the best end
     of the PartialFusion alpha curve and well above NaiveFusion.
"""
import sys
import os
import time
import torch
import numpy as np

sys.path.append("../")
sys.stdout.reconfigure(line_buffering=True)

from src import data_loader
from src.MLP import Deep_MLP
from src.FusionModel.fusion_model import FusionModel
from src.FusionModel.fusion_methods.naive_fusion import NaiveFusion
from src.FusionModel.fusion_methods.partial_fusion import PartialFusion
from src.FusionModel.fusion_methods.git_rebasin import GitRebasin


HIDDEN = 100
ACT = 2
TRAIN_EPOCHS = 5
ALPHAS = [0, 0.2, 0.5, 0.8, 1.0]   # for the PartialFusion comparison column
SAVE_DIR = 'saved'

os.makedirs(SAVE_DIR, exist_ok=True)


def ckpt(which, seed):
    return os.path.join(SAVE_DIR, f'mlp_capacity_{which}_h{HIDDEN}_s{seed}.checkpoint')


def get_or_train(which, seed):
    m = Deep_MLP(hidden_size_1=HIDDEN, hidden_size_2=HIDDEN,
                 hidden_size_3=HIDDEN, which_act=ACT)
    path = ckpt(which, seed)
    if os.path.exists(path):
        try:
            m.load_model(path)
            return m
        except (OSError, RuntimeError, EOFError):
            os.remove(path)
            m = Deep_MLP(hidden_size_1=HIDDEN, hidden_size_2=HIDDEN,
                         hidden_size_3=HIDDEN, which_act=ACT)
    s = 2 * seed if which == 'a' else 2 * seed + 1
    torch.manual_seed(s); torch.cuda.manual_seed(s)
    best, _ = m.train_model_best_ckpt(train_loader, test_loader,
                                      epochs=TRAIN_EPOCHS, verbose=False)
    tmp = path + '.tmp'
    best.save_model(tmp)
    os.replace(tmp, path)
    return best


print("Loading MNIST...")
train_loader, test_loader = data_loader.load_mnist()

print(f"Loading / training two Deep_MLPs (hidden={HIDDEN}, act={ACT})...")
seed = 0
model_a = get_or_train('a', seed)
model_b = get_or_train('b', seed)

acc_a = model_a.test_model(test_loader, criterion=None, verbose=False)
acc_b = model_b.test_model(test_loader, criterion=None, verbose=False)
print(f"  Model A: {acc_a:.2f}%")
print(f"  Model B: {acc_b:.2f}%")


# =====================================================================
# 1) GitRebasin identity checks at lambda=[1,0] and [0,1]
# =====================================================================
print("\n--- Identity check: λ=[1, 0] should reproduce model A ---")
t0 = time.time()
fused = FusionModel(model_a, model_b, GitRebasin(combine_costs=True, pgd=True),
                    lambdas=[1.0, 0.0])
acc = fused.test_model(test_loader, criterion=None, verbose=False)
print(f"  GitRebasin(λ=[1,0]) = {acc:.2f}%  (model A = {acc_a:.2f}%, "
      f"Δ = {acc - acc_a:+.2f})   [{time.time()-t0:.1f}s]")

print("\n--- Identity check: λ=[0, 1] should reproduce model B ---")
t0 = time.time()
fused = FusionModel(model_a, model_b, GitRebasin(combine_costs=True, pgd=True),
                    lambdas=[0.0, 1.0])
acc = fused.test_model(test_loader, criterion=None, verbose=False)
print(f"  GitRebasin(λ=[0,1]) = {acc:.2f}%  (model B = {acc_b:.2f}%, "
      f"Δ = {acc - acc_b:+.2f})   [{time.time()-t0:.1f}s]")


# =====================================================================
# 2) λ=[0.5, 0.5]: compare all three methods
# =====================================================================
print("\n--- Mid-fusion λ=[0.5, 0.5] comparison ---")

t0 = time.time()
naive = FusionModel(model_a, model_b, NaiveFusion(), lambdas=[0.5, 0.5])
acc_naive = naive.test_model(test_loader, criterion=None, verbose=False)
print(f"  NaiveFusion                            = {acc_naive:.2f}%   [{time.time()-t0:.1f}s]")

partial_accs = {}
for a in ALPHAS:
    t0 = time.time()
    fused = FusionModel(model_a, model_b,
                        PartialFusion(alphas=a, combine_costs=True, pgd=True),
                        lambdas=[0.5, 0.5])
    acc = fused.test_model(test_loader, criterion=None, verbose=False)
    partial_accs[a] = acc
    print(f"  PartialFusion α={a:<4}                    = {acc:.2f}%   [{time.time()-t0:.1f}s]")

t0 = time.time()
gr = FusionModel(model_a, model_b, GitRebasin(combine_costs=True, pgd=True),
                 lambdas=[0.5, 0.5])
acc_gr = gr.test_model(test_loader, criterion=None, verbose=False)
print(f"  GitRebasin (Hungarian, no α)           = {acc_gr:.2f}%   [{time.time()-t0:.1f}s]")


# =====================================================================
# 3) Verify the chosen permutation is a valid one-to-one mapping
# =====================================================================
print("\n--- Permutation structure check ---")
gr_method = GitRebasin(combine_costs=True, pgd=True)
_ = FusionModel(model_a, model_b, gr_method, lambdas=[0.5, 0.5])
all_perm = True
for i, k in enumerate(gr_method.kernel_backward):
    if k.size == 0:
        continue
    row_sums = k.sum(axis=1)
    col_sums = k.sum(axis=0)
    is_perm = (
        np.allclose(row_sums[row_sums > 0], 1.0) and
        np.allclose(col_sums[col_sums > 0], 1.0) and
        np.all(np.logical_or(np.isclose(k, 0.0), np.isclose(k, 1.0)))
    )
    print(f"  kernel_backward[{i}]: shape={k.shape}  rowsum∈{{{row_sums.min():.0f},{row_sums.max():.0f}}}  "
          f"colsum∈{{{col_sums.min():.0f},{col_sums.max():.0f}}}  permutation={is_perm}")
    all_perm = all_perm and is_perm
print(f"  → all kernels are 0/1 permutation matrices: {all_perm}")


# =====================================================================
# Summary
# =====================================================================
print("\n=========================== SUMMARY ===========================")
print(f"  Model A                                = {acc_a:.2f}%")
print(f"  Model B                                = {acc_b:.2f}%")
print(f"  NaiveFusion (λ=0.5)                    = {acc_naive:.2f}%")
for a in ALPHAS:
    print(f"  PartialFusion α={a:<4} (λ=0.5)            = {partial_accs[a]:.2f}%")
print(f"  GitRebasin            (λ=0.5)          = {acc_gr:.2f}%")
print("""
Expected:
  - λ=[1,0] / [0,1] identity checks: |Δ| ≲ 0.5 pp from the corresponding model
    (any deviation comes from the permutation being applied + inverted, which
    is exact for floats but not for the surrounding fusion plumbing).
  - At λ=0.5: NaiveFusion << PartialFusion(α=0) ≲ GitRebasin ≈ PartialFusion(α≈0)
    (GitRebasin = Hungarian on the same cost = full alignment, no sink).
  - All kernel_backward matrices should be 0/1 permutations.
""")
