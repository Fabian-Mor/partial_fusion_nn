Reproduction code for the appendix on neuron-level similarities
==================================================================================
We note that major parts of the code for this appendix were coded with assistance
by Claude-Code. We obviously take full responsibility for the correctness.

Requirements:
- Python 3.9+
- PyTorch, torchvision
- matplotlib, numpy

Files:
- train_homogeneous_models.py            : Train two MLPs on MNIST (GELU activation)
- mlp_neuron_similarity_analysis.py      : Analyze MLP neuron similarities
- vgg11_channel_similarity_analysis.py   : Analyze VGG11 channel similarities

Reproduction steps:

1. MLP on MNIST:
   python train_homogeneous_models.py             # Train models (saves to saved_models/)
   python mlp_neuron_similarity_analysis.py       # Generate figures/tables

2. VGG11 on CIFAR10:
   # Requires pre-trained VGG11 checkpoints in saved_models/:
   #   - best_a.checkpoint
   #   - best_b.checkpoint
   # (two arbitrary models from the other VGG11 experiments were taken)
   python vgg11_channel_similarity_analysis.py    # Generate figures/tables


Output directories:
- figures_homogeneous/  : MLP results
- figures_vgg11/        : VGG11 results
