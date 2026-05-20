"""
Analyze neuron similarities for homogeneously trained networks.
Creates publication-quality figures and LaTeX tables.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

# Import model and config from training script
from train_homogeneous_models import (
    MLP, HIDDEN_LAYER_SIZES, NUM_EPOCHS, DEVICE, ACTIVATION,
    get_activation
)

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_ACTIVATION_SAMPLES = 1000  # Number of test samples for activation computation
PERCENTILE_CUTOFF = 80  # Keep this percentage of least isolated neurons
OUTPUT_DIR = "figures_homogeneous"

# =============================================================================
# ACTIVATION EXTRACTION
# =============================================================================

class ActivationExtractor:
    """Extract activations from hidden layers using forward hooks."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.layer_names = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on layers after activation functions."""
        layer_idx = 0
        for i, module in enumerate(self.model.network):
            # Hook after activation functions (ReLU or GELU)
            if isinstance(module, (nn.ReLU, nn.GELU)):
                name = f'layer_{layer_idx}'
                self.layer_names.append(name)
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)
                layer_idx += 1

    def _get_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def get_activations(self, data_loader, num_samples=200):
        """Get activations for a subset of data."""
        self.model.eval()
        all_activations = {name: [] for name in self.layer_names}

        samples_collected = 0
        with torch.no_grad():
            for data, _ in data_loader:
                if samples_collected >= num_samples:
                    break

                batch_size = min(data.size(0), num_samples - samples_collected)
                data = data[:batch_size].to(DEVICE)

                # Forward pass triggers hooks
                _ = self.model(data)

                for name in self.layer_names:
                    all_activations[name].append(self.activations[name].cpu())

                samples_collected += batch_size

        # Concatenate all batches
        for name in self.layer_names:
            all_activations[name] = torch.cat(all_activations[name], dim=0)

        return all_activations

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()


# =============================================================================
# SIMILARITY COMPUTATION
# =============================================================================

def compute_pairwise_distances(activations):
    """
    Compute pairwise Euclidean distances between neurons based on activations.

    Args:
        activations: Tensor of shape (num_samples, num_neurons)

    Returns:
        Distance matrix of shape (num_neurons, num_neurons)
    """
    # Transpose to (num_neurons, num_samples)
    act = activations.T

    # Compute pairwise Euclidean distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_norms = (act ** 2).sum(dim=1, keepdim=True)
    distances_sq = sq_norms + sq_norms.T - 2 * torch.mm(act, act.T)
    distances_sq = torch.clamp(distances_sq, min=0)  # Numerical stability
    distances = torch.sqrt(distances_sq)

    return distances


def compute_cross_network_distances(activations_a, activations_b):
    """
    Compute pairwise distances between neurons from two different networks.

    Args:
        activations_a: Tensor of shape (num_samples, num_neurons_a)
        activations_b: Tensor of shape (num_samples, num_neurons_b)

    Returns:
        Distance matrix of shape (num_neurons_a, num_neurons_b)
    """
    act_a = activations_a.T  # (num_neurons_a, num_samples)
    act_b = activations_b.T  # (num_neurons_b, num_samples)

    sq_norms_a = (act_a ** 2).sum(dim=1, keepdim=True)
    sq_norms_b = (act_b ** 2).sum(dim=1, keepdim=True)

    distances_sq = sq_norms_a + sq_norms_b.T - 2 * torch.mm(act_a, act_b.T)
    distances_sq = torch.clamp(distances_sq, min=0)
    distances = torch.sqrt(distances_sq)

    return distances


def compute_similarity_stats(distances, exclude_self=True):
    """
    Compute nearest neighbor and mean distances for each neuron.

    Args:
        distances: Distance matrix
        exclude_self: If True, exclude diagonal (for within-network comparison)

    Returns:
        nearest_neighbor_distances, mean_distances
    """
    if exclude_self:
        # Set diagonal to infinity to exclude self-distances
        mask = torch.eye(distances.size(0), dtype=torch.bool)
        distances = distances.clone()
        distances[mask] = float('inf')

    nearest_neighbor = distances.min(dim=1)[0]

    if exclude_self:
        # For mean, exclude diagonal
        distances[mask] = 0
        mean_dist = distances.sum(dim=1) / (distances.size(1) - 1)
    else:
        mean_dist = distances.mean(dim=1)

    return nearest_neighbor.numpy(), mean_dist.numpy()


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_publication_figure(stats_dict, output_path):
    """Create publication-quality histogram figure."""

    # Publication settings
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.family': 'serif',
    })

    # Colorblind-friendly colors
    color_a = '#0072B2'  # Blue
    color_b = '#D55E00'  # Vermillion

    num_layers = len([k for k in stats_dict.keys() if k.startswith('layer_')])

    fig, axes = plt.subplots(num_layers, 4, figsize=(10, 2.2 * num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)

    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'
        stats = stats_dict[layer_key]

        # Column titles (only on first row)
        if layer_idx == 0:
            axes[0, 0].set_title('NN Dist (Within)', fontweight='bold')
            axes[0, 1].set_title('NN Dist (Cross)', fontweight='bold')
            axes[0, 2].set_title('Mean Dist (Within)', fontweight='bold')
            axes[0, 3].set_title('Mean Dist (Cross)', fontweight='bold')

        # Row labels
        axes[layer_idx, 0].set_ylabel(f'Layer {layer_idx + 1}')

        # Nearest neighbor within
        ax = axes[layer_idx, 0]
        ax.hist(stats['nn_within_a'], bins=20, alpha=0.7, color=color_a, label='Model A', density=True)
        ax.hist(stats['nn_within_b'], bins=20, alpha=0.7, color=color_b, label='Model B', density=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if layer_idx == 0:
            ax.legend(loc='upper right', frameon=False)

        # Nearest neighbor cross
        ax = axes[layer_idx, 1]
        ax.hist(stats['nn_cross_a_to_b'], bins=20, alpha=0.7, color=color_a, label='A→B', density=True)
        ax.hist(stats['nn_cross_b_to_a'], bins=20, alpha=0.7, color=color_b, label='B→A', density=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if layer_idx == 0:
            ax.legend(loc='upper right', frameon=False)

        # Mean distance within
        ax = axes[layer_idx, 2]
        ax.hist(stats['mean_within_a'], bins=20, alpha=0.7, color=color_a, density=True)
        ax.hist(stats['mean_within_b'], bins=20, alpha=0.7, color=color_b, density=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Mean distance cross
        ax = axes[layer_idx, 3]
        ax.hist(stats['mean_cross_a_to_b'], bins=20, alpha=0.7, color=color_a, density=True)
        ax.hist(stats['mean_cross_b_to_a'], bins=20, alpha=0.7, color=color_b, density=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save as PDF (vector) and PNG (preview)
    plt.savefig(output_path.replace('.pdf', '.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Saved figure to {output_path}")


def create_latex_table(stats_dict, output_path, percentile=80):
    """Create LaTeX table with summary statistics."""

    num_layers = len([k for k in stats_dict.keys() if k.startswith('layer_')])

    def get_percentile_mean(arr, percentile):
        """Get mean of the lowest percentile of values."""
        threshold = np.percentile(arr, percentile)
        return np.mean(arr[arr <= threshold])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Neuron similarity statistics for homogeneously trained networks (GELU activation).")
    lines.append(r"\textbf{Top:} Mean distance averaged over all neurons.")
    lines.append(r"\textbf{Middle:} Mean distance averaged over the " + str(percentile) + r"\% of neurons with smallest distances.")
    lines.append(r"\textbf{Bottom:} Difference, quantifying the contribution of the " + str(100-percentile) + r"\% most isolated neurons.")
    lines.append(r"Both models trained on the full MNIST training set.}")
    lines.append(r"\label{tab:neuron_similarity_homogeneous}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l cccc cccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{4}{c}{\textbf{Nearest Neighbor Distance}} & \multicolumn{4}{c}{\textbf{Mean Distance to All}} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"& \multicolumn{2}{c}{Within} & \multicolumn{2}{c}{Across} & \multicolumn{2}{c}{Within} & \multicolumn{2}{c}{Across} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"Layer & A & B & A{\footnotesize$\to$}B & B{\footnotesize$\to$}A & A & B & A{\footnotesize$\to$}B & B{\footnotesize$\to$}A \\")
    lines.append(r"\midrule")

    # All neurons
    lines.append(r"\multicolumn{9}{@{}l}{\textit{All neurons}} \\")
    for layer_idx in range(num_layers):
        stats = stats_dict[f'layer_{layer_idx}']
        line = f"{layer_idx + 1} & "
        line += f"{np.mean(stats['nn_within_a']):.1f} & {np.mean(stats['nn_within_b']):.1f} & "
        line += f"{np.mean(stats['nn_cross_a_to_b']):.1f} & {np.mean(stats['nn_cross_b_to_a']):.1f} & "
        line += f"{np.mean(stats['mean_within_a']):.1f} & {np.mean(stats['mean_within_b']):.1f} & "
        line += f"{np.mean(stats['mean_cross_a_to_b']):.1f} & {np.mean(stats['mean_cross_b_to_a']):.1f} \\\\"
        lines.append(line)

    lines.append(r"\midrule")

    # Percentile filtered
    lines.append(r"\multicolumn{9}{@{}l}{\textit{" + str(percentile) + r"\% least isolated neurons}} \\")
    for layer_idx in range(num_layers):
        stats = stats_dict[f'layer_{layer_idx}']
        line = f"{layer_idx + 1} & "
        line += f"{get_percentile_mean(stats['nn_within_a'], percentile):.1f} & "
        line += f"{get_percentile_mean(stats['nn_within_b'], percentile):.1f} & "
        line += f"{get_percentile_mean(stats['nn_cross_a_to_b'], percentile):.1f} & "
        line += f"{get_percentile_mean(stats['nn_cross_b_to_a'], percentile):.1f} & "
        line += f"{get_percentile_mean(stats['mean_within_a'], percentile):.1f} & "
        line += f"{get_percentile_mean(stats['mean_within_b'], percentile):.1f} & "
        line += f"{get_percentile_mean(stats['mean_cross_a_to_b'], percentile):.1f} & "
        line += f"{get_percentile_mean(stats['mean_cross_b_to_a'], percentile):.1f} \\\\"
        lines.append(line)

    lines.append(r"\midrule")

    # Difference (isolated neuron contribution)
    lines.append(r"\multicolumn{9}{@{}l}{\textit{Difference (isolated neuron contribution)}} \\")
    for layer_idx in range(num_layers):
        stats = stats_dict[f'layer_{layer_idx}']
        line = f"{layer_idx + 1} & "

        diff_nn_a = np.mean(stats['nn_within_a']) - get_percentile_mean(stats['nn_within_a'], percentile)
        diff_nn_b = np.mean(stats['nn_within_b']) - get_percentile_mean(stats['nn_within_b'], percentile)
        diff_nn_ab = np.mean(stats['nn_cross_a_to_b']) - get_percentile_mean(stats['nn_cross_a_to_b'], percentile)
        diff_nn_ba = np.mean(stats['nn_cross_b_to_a']) - get_percentile_mean(stats['nn_cross_b_to_a'], percentile)
        diff_mean_a = np.mean(stats['mean_within_a']) - get_percentile_mean(stats['mean_within_a'], percentile)
        diff_mean_b = np.mean(stats['mean_within_b']) - get_percentile_mean(stats['mean_within_b'], percentile)
        diff_mean_ab = np.mean(stats['mean_cross_a_to_b']) - get_percentile_mean(stats['mean_cross_a_to_b'], percentile)
        diff_mean_ba = np.mean(stats['mean_cross_b_to_a']) - get_percentile_mean(stats['mean_cross_b_to_a'], percentile)

        line += f"{diff_nn_a:.1f} & {diff_nn_b:.1f} & {diff_nn_ab:.1f} & {diff_nn_ba:.1f} & "
        line += f"{diff_mean_a:.1f} & {diff_mean_b:.1f} & {diff_mean_ab:.1f} & {diff_mean_ba:.1f} \\\\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  Saved table to {output_path}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def get_model_filename(model_name):
    """Generate filename based on model parameters."""
    hidden_str = "_".join(map(str, HIDDEN_LAYER_SIZES))
    return f"saved_models/{model_name}_homogeneous_{ACTIVATION}_h{hidden_str}_e{NUM_EPOCHS}.pt"


def load_model(model_name):
    """Load model from disk."""
    filename = get_model_filename(model_name)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model not found: {filename}. Run train_homogeneous_models.py first.")

    checkpoint = torch.load(filename, map_location=DEVICE)
    model = MLP(hidden_sizes=checkpoint['hidden_sizes'],
                activation=checkpoint.get('activation', 'relu')).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():
    print("=" * 60)
    print("Neuron Similarity Analysis - Homogeneous Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Activation samples: {NUM_ACTIVATION_SAMPLES}")
    print(f"  Percentile cutoff: {PERCENTILE_CUTOFF}%")
    print(f"  Activation function: {ACTIVATION}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models
    print("\nLoading models...")
    model_a = load_model("model_a")
    model_b = load_model("model_b")
    print("  Models loaded successfully")

    # Prepare test data
    print("\nPreparing test data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Extract activations
    print("\nExtracting activations...")
    extractor_a = ActivationExtractor(model_a)
    extractor_b = ActivationExtractor(model_b)

    activations_a = extractor_a.get_activations(test_loader, NUM_ACTIVATION_SAMPLES)
    activations_b = extractor_b.get_activations(test_loader, NUM_ACTIVATION_SAMPLES)

    extractor_a.remove_hooks()
    extractor_b.remove_hooks()

    num_layers = len(activations_a)
    print(f"  Extracted activations from {num_layers} layers")

    # Compute statistics for each layer
    print("\nComputing similarity statistics...")
    stats_dict = {}

    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'
        act_a = activations_a[layer_key]
        act_b = activations_b[layer_key]

        # Within-network distances
        dist_within_a = compute_pairwise_distances(act_a)
        dist_within_b = compute_pairwise_distances(act_b)

        # Cross-network distances
        dist_cross = compute_cross_network_distances(act_a, act_b)

        # Compute stats
        nn_within_a, mean_within_a = compute_similarity_stats(dist_within_a, exclude_self=True)
        nn_within_b, mean_within_b = compute_similarity_stats(dist_within_b, exclude_self=True)

        # For cross-network: A->B means for each neuron in A, find NN in B
        nn_cross_a_to_b = dist_cross.min(dim=1)[0].numpy()
        mean_cross_a_to_b = dist_cross.mean(dim=1).numpy()

        # B->A means for each neuron in B, find NN in A
        nn_cross_b_to_a = dist_cross.min(dim=0)[0].numpy()
        mean_cross_b_to_a = dist_cross.mean(dim=0).numpy()

        stats_dict[layer_key] = {
            'nn_within_a': nn_within_a,
            'nn_within_b': nn_within_b,
            'nn_cross_a_to_b': nn_cross_a_to_b,
            'nn_cross_b_to_a': nn_cross_b_to_a,
            'mean_within_a': mean_within_a,
            'mean_within_b': mean_within_b,
            'mean_cross_a_to_b': mean_cross_a_to_b,
            'mean_cross_b_to_a': mean_cross_b_to_a,
        }

        print(f"  Layer {layer_idx + 1}: NN within A={np.mean(nn_within_a):.2f}, "
              f"NN within B={np.mean(nn_within_b):.2f}, "
              f"NN cross A→B={np.mean(nn_cross_a_to_b):.2f}")

    # Create outputs
    print("\nGenerating outputs...")
    create_publication_figure(stats_dict, f"{OUTPUT_DIR}/neuron_similarity_histograms_homogeneous.pdf")
    create_latex_table(stats_dict, f"{OUTPUT_DIR}/neuron_similarity_table_homogeneous.tex", PERCENTILE_CUTOFF)

    print("\nDone!")


if __name__ == "__main__":
    main()
