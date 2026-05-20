"""
Analyze channel similarities for VGG11 networks trained on CIFAR10.

For CNNs, we compute channel-wise similarity by:
1. Extracting activations after each ReLU layer (shape: batch x channels x H x W)
2. For each channel, computing its mean activation over spatial dimensions (H x W)
3. This gives a feature vector of shape (num_samples,) per channel
4. Computing pairwise Euclidean distances between channel feature vectors
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use small number for proof-of-concept (no GPU = slow)
NUM_ACTIVATION_SAMPLES = 1000  # Keep small for CPU
PERCENTILE_CUTOFF = 80
OUTPUT_DIR = "figures_vgg11"
DEVICE = torch.device('cpu')  # Force CPU

# =============================================================================
# VGG11 MODEL DEFINITION
# =============================================================================

class VGG11(nn.Module):
    """VGG11 model matching the checkpoint format."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Channel sizes for standard VGG11
        channel_sizes = [64, 128, 256, 256, 512, 512, 512, 512]

        # Conv block 1
        self.conv1 = nn.Conv2d(3, channel_sizes[0], kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2
        self.conv2 = nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3
        self.conv3 = nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4
        self.conv5 = nn.Conv2d(channel_sizes[3], channel_sizes[4], kernel_size=3, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(channel_sizes[4], channel_sizes[5], kernel_size=3, padding=1, bias=False)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 5
        self.conv7 = nn.Conv2d(channel_sizes[5], channel_sizes[6], kernel_size=3, padding=1, bias=False)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(channel_sizes[6], channel_sizes[7], kernel_size=3, padding=1, bias=False)
        self.relu8 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(channel_sizes[7], num_classes, bias=False)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.maxpool3(self.relu4(self.conv4(x)))
        x = self.relu5(self.conv5(x))
        x = self.maxpool4(self.relu6(self.conv6(x)))
        x = self.relu7(self.conv7(x))
        x = self.maxpool5(self.relu8(self.conv8(x)))
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def load_vgg11_checkpoint(filepath):
    """Load VGG11 model from checkpoint with key remapping."""
    model = VGG11()

    checkpoint = torch.load(filepath, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]

    # Map old keys to new keys
    key_mapping = {
        "features.0.weight": "conv1.weight",
        "features.3.weight": "conv2.weight",
        "features.6.weight": "conv3.weight",
        "features.8.weight": "conv4.weight",
        "features.11.weight": "conv5.weight",
        "features.13.weight": "conv6.weight",
        "features.16.weight": "conv7.weight",
        "features.18.weight": "conv8.weight",
        "classifier.weight": "classifier.weight",
    }

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = key_mapping.get(k, k)
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# =============================================================================
# ACTIVATION EXTRACTION FOR CNNs
# =============================================================================

class ChannelActivationExtractor:
    """
    Extract channel-wise activations from CNN layers.

    For each channel, we compute the mean activation over spatial dimensions,
    giving a feature vector of shape (num_samples,) per channel.
    """

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.layer_names = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on ReLU layers after conv layers."""
        relu_layers = [
            ('relu1', self.model.relu1),
            ('relu2', self.model.relu2),
            ('relu3', self.model.relu3),
            ('relu4', self.model.relu4),
            ('relu5', self.model.relu5),
            ('relu6', self.model.relu6),
            ('relu7', self.model.relu7),
            ('relu8', self.model.relu8),
        ]

        for name, layer in relu_layers:
            self.layer_names.append(name)
            hook = layer.register_forward_hook(self._get_hook(name))
            self.hooks.append(hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            # output shape: (batch, channels, H, W)
            # Compute mean over spatial dimensions for each channel
            # Result shape: (batch, channels)
            channel_means = output.mean(dim=(2, 3)).detach()
            self.activations[name] = channel_means
        return hook

    def get_activations(self, data_loader, num_samples=50):
        """Get channel activations for a subset of data."""
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
                if samples_collected % 10 == 0:
                    print(f"    Collected {samples_collected}/{num_samples} samples")

        # Concatenate all batches: shape (num_samples, num_channels)
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
    Compute pairwise Euclidean distances between channels.

    Args:
        activations: Tensor of shape (num_samples, num_channels)

    Returns:
        Distance matrix of shape (num_channels, num_channels)
    """
    # Transpose to (num_channels, num_samples)
    act = activations.T

    # Compute pairwise Euclidean distances
    sq_norms = (act ** 2).sum(dim=1, keepdim=True)
    distances_sq = sq_norms + sq_norms.T - 2 * torch.mm(act, act.T)
    distances_sq = torch.clamp(distances_sq, min=0)
    distances = torch.sqrt(distances_sq)

    return distances


def compute_cross_network_distances(activations_a, activations_b):
    """
    Compute pairwise distances between channels from two different networks.

    Args:
        activations_a: Tensor of shape (num_samples, num_channels_a)
        activations_b: Tensor of shape (num_samples, num_channels_b)

    Returns:
        Distance matrix of shape (num_channels_a, num_channels_b)
    """
    act_a = activations_a.T
    act_b = activations_b.T

    sq_norms_a = (act_a ** 2).sum(dim=1, keepdim=True)
    sq_norms_b = (act_b ** 2).sum(dim=1, keepdim=True)

    distances_sq = sq_norms_a + sq_norms_b.T - 2 * torch.mm(act_a, act_b.T)
    distances_sq = torch.clamp(distances_sq, min=0)
    distances = torch.sqrt(distances_sq)

    return distances


def compute_similarity_stats(distances, exclude_self=True):
    """
    Compute nearest neighbor and mean distances for each channel.

    Args:
        distances: Distance matrix
        exclude_self: If True, exclude diagonal (for within-network comparison)

    Returns:
        nearest_neighbor_distances, mean_distances
    """
    if exclude_self:
        mask = torch.eye(distances.size(0), dtype=torch.bool)
        distances = distances.clone()
        distances[mask] = float('inf')

    nearest_neighbor = distances.min(dim=1)[0]

    if exclude_self:
        distances[mask] = 0
        mean_dist = distances.sum(dim=1) / (distances.size(1) - 1)
    else:
        mean_dist = distances.mean(dim=1)

    return nearest_neighbor.numpy(), mean_dist.numpy()


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_publication_figure(stats_dict, output_path):
    """Create publication-quality histogram figure for channel similarities."""

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

    color_a = '#0072B2'
    color_b = '#D55E00'

    layer_names = [k for k in stats_dict.keys()]
    num_layers = len(layer_names)

    fig, axes = plt.subplots(num_layers, 4, figsize=(10, 2.0 * num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)

    for layer_idx, layer_key in enumerate(layer_names):
        stats = stats_dict[layer_key]
        num_channels = len(stats['nn_within_a'])

        if layer_idx == 0:
            axes[0, 0].set_title('NN Dist (Within)', fontweight='bold')
            axes[0, 1].set_title('NN Dist (Cross)', fontweight='bold')
            axes[0, 2].set_title('Mean Dist (Within)', fontweight='bold')
            axes[0, 3].set_title('Mean Dist (Cross)', fontweight='bold')

        axes[layer_idx, 0].set_ylabel(f'{layer_key}\n({num_channels} ch)')

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
    plt.savefig(output_path.replace('.pdf', '.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Saved figure to {output_path}")


def create_latex_table(stats_dict, output_path, percentile=80):
    """Create LaTeX table with summary statistics."""

    layer_names = list(stats_dict.keys())

    def get_percentile_mean(arr, percentile):
        threshold = np.percentile(arr, percentile)
        return np.mean(arr[arr <= threshold])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Channel similarity statistics for VGG11 on CIFAR10.")
    lines.append(r"\textbf{Top:} Mean distance averaged over all channels.")
    lines.append(r"\textbf{Middle:} Mean distance averaged over the " + str(percentile) + r"\% of channels with smallest distances.")
    lines.append(r"\textbf{Bottom:} Difference (isolated channel contribution).}")
    lines.append(r"\label{tab:channel_similarity_vgg11}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l cccc cccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{4}{c}{\textbf{Nearest Neighbor Distance}} & \multicolumn{4}{c}{\textbf{Mean Distance to All}} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"& \multicolumn{2}{c}{Within} & \multicolumn{2}{c}{Across} & \multicolumn{2}{c}{Within} & \multicolumn{2}{c}{Across} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"Layer & A & B & A{\footnotesize$\to$}B & B{\footnotesize$\to$}A & A & B & A{\footnotesize$\to$}B & B{\footnotesize$\to$}A \\")
    lines.append(r"\midrule")

    # All channels
    lines.append(r"\multicolumn{9}{@{}l}{\textit{All channels}} \\")
    for layer_key in layer_names:
        stats = stats_dict[layer_key]
        # Use short layer name (e.g., "relu1" -> "1")
        layer_num = layer_key.replace('relu', '')
        line = f"{layer_num} & "
        line += f"{np.mean(stats['nn_within_a']):.2f} & {np.mean(stats['nn_within_b']):.2f} & "
        line += f"{np.mean(stats['nn_cross_a_to_b']):.2f} & {np.mean(stats['nn_cross_b_to_a']):.2f} & "
        line += f"{np.mean(stats['mean_within_a']):.2f} & {np.mean(stats['mean_within_b']):.2f} & "
        line += f"{np.mean(stats['mean_cross_a_to_b']):.2f} & {np.mean(stats['mean_cross_b_to_a']):.2f} \\\\"
        lines.append(line)

    lines.append(r"\midrule")

    # Percentile filtered
    lines.append(r"\multicolumn{9}{@{}l}{\textit{" + str(percentile) + r"\% least isolated channels}} \\")
    for layer_key in layer_names:
        stats = stats_dict[layer_key]
        layer_num = layer_key.replace('relu', '')
        line = f"{layer_num} & "
        line += f"{get_percentile_mean(stats['nn_within_a'], percentile):.2f} & "
        line += f"{get_percentile_mean(stats['nn_within_b'], percentile):.2f} & "
        line += f"{get_percentile_mean(stats['nn_cross_a_to_b'], percentile):.2f} & "
        line += f"{get_percentile_mean(stats['nn_cross_b_to_a'], percentile):.2f} & "
        line += f"{get_percentile_mean(stats['mean_within_a'], percentile):.2f} & "
        line += f"{get_percentile_mean(stats['mean_within_b'], percentile):.2f} & "
        line += f"{get_percentile_mean(stats['mean_cross_a_to_b'], percentile):.2f} & "
        line += f"{get_percentile_mean(stats['mean_cross_b_to_a'], percentile):.2f} \\\\"
        lines.append(line)

    lines.append(r"\midrule")

    # Difference
    lines.append(r"\multicolumn{9}{@{}l}{\textit{Difference (isolated channel contribution)}} \\")
    for layer_key in layer_names:
        stats = stats_dict[layer_key]
        layer_num = layer_key.replace('relu', '')
        line = f"{layer_num} & "

        diff_nn_a = np.mean(stats['nn_within_a']) - get_percentile_mean(stats['nn_within_a'], percentile)
        diff_nn_b = np.mean(stats['nn_within_b']) - get_percentile_mean(stats['nn_within_b'], percentile)
        diff_nn_ab = np.mean(stats['nn_cross_a_to_b']) - get_percentile_mean(stats['nn_cross_a_to_b'], percentile)
        diff_nn_ba = np.mean(stats['nn_cross_b_to_a']) - get_percentile_mean(stats['nn_cross_b_to_a'], percentile)
        diff_mean_a = np.mean(stats['mean_within_a']) - get_percentile_mean(stats['mean_within_a'], percentile)
        diff_mean_b = np.mean(stats['mean_within_b']) - get_percentile_mean(stats['mean_within_b'], percentile)
        diff_mean_ab = np.mean(stats['mean_cross_a_to_b']) - get_percentile_mean(stats['mean_cross_a_to_b'], percentile)
        diff_mean_ba = np.mean(stats['mean_cross_b_to_a']) - get_percentile_mean(stats['mean_cross_b_to_a'], percentile)

        line += f"{diff_nn_a:.2f} & {diff_nn_b:.2f} & {diff_nn_ab:.2f} & {diff_nn_ba:.2f} & "
        line += f"{diff_mean_a:.2f} & {diff_mean_b:.2f} & {diff_mean_ab:.2f} & {diff_mean_ba:.2f} \\\\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  Saved table to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Channel Similarity Analysis - VGG11 on CIFAR10")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Activation samples: {NUM_ACTIVATION_SAMPLES}")
    print(f"  Percentile cutoff: {PERCENTILE_CUTOFF}%")
    print(f"  Device: {DEVICE}")
    print(f"  Output directory: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models
    print("\nLoading models...")
    model_a = load_vgg11_checkpoint("saved_models/best_a.checkpoint")
    model_b = load_vgg11_checkpoint("saved_models/best_b.checkpoint")
    print("  Models loaded successfully")

    # Prepare CIFAR10 test data
    print("\nPreparing CIFAR10 test data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Extract activations
    print("\nExtracting activations from Model A...")
    extractor_a = ChannelActivationExtractor(model_a)
    activations_a = extractor_a.get_activations(test_loader, NUM_ACTIVATION_SAMPLES)
    extractor_a.remove_hooks()

    print("\nExtracting activations from Model B...")
    extractor_b = ChannelActivationExtractor(model_b)
    activations_b = extractor_b.get_activations(test_loader, NUM_ACTIVATION_SAMPLES)
    extractor_b.remove_hooks()

    layer_names = list(activations_a.keys())
    print(f"\n  Extracted activations from {len(layer_names)} layers")
    for name in layer_names:
        print(f"    {name}: {activations_a[name].shape[1]} channels")

    # Compute statistics
    print("\nComputing similarity statistics...")
    stats_dict = {}

    for layer_key in layer_names:
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

        nn_cross_a_to_b = dist_cross.min(dim=1)[0].numpy()
        mean_cross_a_to_b = dist_cross.mean(dim=1).numpy()
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

        print(f"  {layer_key}: NN within A={np.mean(nn_within_a):.3f}, "
              f"NN within B={np.mean(nn_within_b):.3f}, "
              f"NN cross A→B={np.mean(nn_cross_a_to_b):.3f}")

    # Generate outputs
    print("\nGenerating outputs...")
    create_publication_figure(stats_dict, f"{OUTPUT_DIR}/channel_similarity_histograms_vgg11.pdf")
    create_latex_table(stats_dict, f"{OUTPUT_DIR}/channel_similarity_table_vgg11.tex", PERCENTILE_CUTOFF)

    print("\nDone!")


if __name__ == "__main__":
    main()
