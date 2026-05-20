# Partial Fusion of Neural Networks

Reference implementation accompanying the paper *Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation (Morelli & Eckstein, ICML 2026)*. The repository implements **partial fusion**, a method for merging trained neural networks using partial optimal transport with a sink option.

## Installation

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate partial_fusion
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: Python 3.11, PyTorch ≥ 2.6, CUDA-capable GPU recommended for the CNN/ResNet experiments.

## Datasets

MNIST and CIFAR-10 are downloaded automatically by `torchvision` the first time you run any experiment. No manual data setup is required.

## Pretrained checkpoints

The fusion and pruning experiments load pre-trained source models. Re-training all of them from scratch takes a long time on a single GPU, so we host the full set of checkpoints (~1.1 GB) on Zenodo:

> **Download:** <https://zenodo.org/records/20306973> (`partial_fusion_checkpoints.tar.gz`, ~995 MB)
> **SHA-256:** `28a8a6818803085056dc70c903689ed8bc4e1690fc4803931ee1d242e8f31156`

After downloading `partial_fusion_checkpoints.tar.gz`, extract it inside the `experiments/` directory so that its contents end up at `experiments/saved/` and `experiments/saved_models/`:

```bash
cd experiments
tar -xzvf /path/to/partial_fusion_checkpoints.tar.gz
```

The archive is laid out so that the file paths inside it match exactly what the experiment scripts expect to load:

- **`saved/resnet18_a_{0..10}.checkpoint`** / **`resnet18_b_{0..10}.checkpoint`** — 11 independently-seeded ResNet18 pairs on CIFAR-10 used by `cifar_resnet18_partial_fusion.py`.
- **`saved/resnet18_evenodd_{a,b}.checkpoint`** — class-split specialists for the two `cifar_resnet18_class_split_*.py` scripts.
- **`saved/mlp_capacity_{a,b}_h{25,50,100,200,400}_s{0..4}.checkpoint`** — 50 Deep_MLPs across the full capacity sweep used by `mnist_mlp_capacity_sweep.py` (and seed 0 / hidden 100 is also used by `mnist_mlp_git_rebasin_baseline.py`).
- **`saved_models/best_{a,b}.checkpoint`** — VGG11 checkpoints used by the appendix channel-similarity analysis.

If you'd rather train everything yourself, skip the download and run the pretraining utilities in `experiments/` (see the section below); the experiment scripts also train missing source models on demand.

## Project layout

```
src/
  base_model.py                 # BaseModel: training/eval/save/load helpers shared by all architectures
  data_loader.py                # MNIST / CIFAR-10 loaders, including the specialist (digit-biased) split
  CNN/                          # ResNet18, VGG11
  MLP/                          # MLP, Deep_MLP
  FusionModel/
    fusion_model.py             # FusionModel: assembles a forward network from aligned weights
    fusion_methods/
      base_fusion.py            # Abstract fusion-method interface
      naive_fusion.py           # Lambda-weighted weight averaging (baseline)
      partial_fusion.py         # Partial OT alignment with sink option (this paper's method)
      git_rebasin.py            # Git Re-Basin baseline
    generalized_pruning/        # Structured, weight-hierarchical, and stochastic-hierarchical pruning
experiments/
  # --- Pretraining utilities (produce checkpoints used by downstream experiments) ---
  mnist_mlp_train_baselines.py
  cifar_vgg11_train_baselines.py
  mnist_mlp_train_specialists.py
  # --- Main partial fusion experiments ---
  partial_fusion_sweep.py
  cifar_resnet18_partial_fusion.py
  # --- Class-split fusion ablations (ResNet18 + CIFAR-10) ---
  cifar_resnet18_class_split_cosine_bn.py
  cifar_resnet18_class_split_random_widen.py
  # --- MLP capacity ablation ---
  mnist_mlp_capacity_sweep.py
  # --- Baseline method comparison ---
  mnist_mlp_git_rebasin_baseline.py
  # --- Pruning experiments ---
  single_model_pruning_sweep.py
  mnist_mlp_seed_pruning.py
  cifar_vgg11_seed_pruning.py
  mnist_mlp_ensemble_pruning.py
  cifar_vgg11_ensemble_pruning.py
  # --- Paper appendix (neuron / channel similarity analysis) ---
  appendix_experiments/
```

All experiment scripts are designed to be run from inside `experiments/`:

```bash
cd experiments
python <script_name>.py
```

## Experiments

### Pretraining utilities

These scripts produce the model checkpoints that the downstream fusion and pruning experiments load. They write to `experiments/saved_compression/` (or, for the ResNet18 experiments, to `experiments/saved/`).

- **`mnist_mlp_train_baselines.py`** — Trains `Deep_MLP` models on a 10% subset of MNIST across uniform hidden widths (100, 120, …, 200), producing single-model baselines and characterising the MLP capacity–accuracy curve.
- **`cifar_vgg11_train_baselines.py`** — Trains VGG11 models on CIFAR-10 with channel-width multipliers from 0.9× down to 0.1×, producing the single-model VGG11 reference checkpoints used by the pruning experiments.
- **`mnist_mlp_train_specialists.py`** — Trains pairs of `Deep_MLP` models per seed and activation: one on the full MNIST training set (the *general* model) and one biased toward digit 4 (the *specific* model). These pairs feed the specialist-fusion and ensemble-pruning experiments.

### Main partial fusion experiments

- **`partial_fusion_sweep.py`** — The core experiment. Loads two pre-trained models (either a specialist pair or two independently-seeded models) and sweeps over `lambda` (mixing weight) and `alpha` (sink mass). Reports accuracy and effective parameter counts for naive averaging, partial OT fusion, and several pruning-based variants. Toggle `CNN = True` for VGG11+CIFAR-10 or `False` for MLP+MNIST.
- **`cifar_resnet18_partial_fusion.py`** — ResNet18 + CIFAR-10 instance of the same sweep, run over `num_seeds` independently-trained model pairs (checkpoints at `saved/resnet18_a_{i}.checkpoint` / `saved/resnet18_b_{i}.checkpoint`). Setting `save=True` trains the pairs from scratch (standard SGD recipe: lr=0.1, momentum=0.9, weight_decay=5e-4, cosine annealing, 200 epochs); otherwise existing checkpoints are loaded and any missing pair is trained on demand.

### Class-split fusion ablations (ResNet18 + CIFAR-10)

Both scripts use the same data split: model A is trained primarily on even classes (`{0,2,4,6,8}`) and model B on odd classes (`{1,3,5,7,9}`), with a small fraction of cross-class data for fine-tuning. If `saved/resnet18_evenodd_{a,b}.checkpoint` does not exist, the scripts train the source models from scratch.

- **`cifar_resnet18_class_split_cosine_bn.py`** — Tests cosine-similarity initialization of batch-norm running statistics during fusion (as opposed to recalibration on a held-out batch), then masked-fine-tunes each fused configuration across `alpha` values.
- **`cifar_resnet18_class_split_random_widen.py`** — Sanity check: widens model A with random extra channels to match the parameter count of the partially fused model, fine-tunes it, and compares against partial fusion. Isolates whether the fusion gains come from genuine cross-model knowledge transfer or simply from added capacity.

### MLP capacity ablation

- **`mnist_mlp_capacity_sweep.py`** — Outer axis is the uniform MLP hidden width (e.g., `[200, 400]`). For each capacity, trains five seed pairs and runs a full `(lambda, alpha)` fusion sweep. Reports mean ± standard deviation across seeds, isolating how the partial-fusion benefit scales with model size.

### Baseline method comparison

- **`mnist_mlp_git_rebasin_baseline.py`** — Sanity check + comparison against the Git Re-Basin baseline. Verifies the identity-recovery property (`lambda=[1,0]` reproduces model A, `lambda=[0,1]` reproduces model B), then reports test accuracy at `lambda=[0.5,0.5]` for `NaiveFusion`, `PartialFusion` across alphas, and `GitRebasin`.

### Pruning experiments

- **`single_model_pruning_sweep.py`** — Configurable single-model pruning experiment (MLP+MNIST or VGG11+CIFAR-10). Compares structured pruning, optimal-transport pruning, and stochastic hierarchical pruning across width multipliers, with optional post-pruning fine-tuning.
- **`mnist_mlp_seed_pruning.py`** — Pruning comparison on independently-seeded MLPs (10 seeds, width multipliers from 1.0× down to 0.1×). Compares simple structured pruning, post-processed pruning, and activation-based clustering, reporting mean ± std test accuracy per ratio.
- **`cifar_vgg11_seed_pruning.py`** — VGG11 + CIFAR-10 counterpart of the above. Compares simple incoming-importance pruning, OT-based pruning ("ot paper"), and stochastic-hierarchical clustering on independently-seeded models.
- **`mnist_mlp_ensemble_pruning.py`** — Combines partial fusion with pruning on the specialist split: fuses general + specific MLPs across `alpha`, then prunes the fused model and compares the resulting accuracy across pruning methods.
- **`cifar_vgg11_ensemble_pruning.py`** — Same idea on VGG11 + CIFAR-10 using different-seed ensemble pairs.

### Paper appendix — neuron / channel similarity analysis

See `experiments/appendix_experiments/README.txt` for the appendix reproduction notes.

- **`train_homogeneous_models.py`** — Trains two `MLP`s on the full MNIST training set with the GELU activation, producing the homogeneous baseline checkpoints used by `mlp_neuron_similarity_analysis.py`.
- **`mlp_neuron_similarity_analysis.py`** — Computes pairwise neuron-level activation distances between the two homogeneously-trained MLPs and produces the figures and LaTeX tables included in the appendix.
- **`vgg11_channel_similarity_analysis.py`** — Computes per-channel activation distances between two VGG11 checkpoints (any two from the VGG11 experiments) and produces the corresponding appendix figures.

## Reproduction order

A typical end-to-end reproduction proceeds in three phases:

```bash
cd experiments

# 1. Pretrain the source models.
python mnist_mlp_train_baselines.py
python cifar_vgg11_train_baselines.py
python mnist_mlp_train_specialists.py

# 2. Run the main fusion experiments and ablations.
python partial_fusion_sweep.py
python cifar_resnet18_partial_fusion.py
python cifar_resnet18_class_split_cosine_bn.py
python cifar_resnet18_class_split_random_widen.py
python mnist_mlp_capacity_sweep.py
python mnist_mlp_git_rebasin_baseline.py

# 3. Run the pruning experiments.
python single_model_pruning_sweep.py
python mnist_mlp_seed_pruning.py
python cifar_vgg11_seed_pruning.py
python mnist_mlp_ensemble_pruning.py
python cifar_vgg11_ensemble_pruning.py
```

The appendix can be reproduced separately:

```bash
cd experiments/appendix_experiments
python train_homogeneous_models.py
python mlp_neuron_similarity_analysis.py
python vgg11_channel_similarity_analysis.py
```

`cifar_resnet18_partial_fusion.py` and the `cifar_resnet18_class_split_*` scripts do not depend on the pretraining utilities: they either train their own ResNet18 checkpoints on first run or load existing ones from `saved/`.


## License

This project is released under the [MIT License](LICENSE).
