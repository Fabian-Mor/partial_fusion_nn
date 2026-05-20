import sys
sys.path.append("../")
from src.MLP import Deep_MLP
from src import data_loader

train_loader, _ = data_loader.load_mnist_subset(percentage=0.1)
_, test_loader = data_loader.load_mnist()
act = 2
hidden_sizes = [[100, 100, 100], #0.01: 88.06, 0.03: 91.84, 0.05: 92.74, 0.1: 95.78
                [120, 120, 120], #0.01: 87.46, 0.03: 92.10, 0.05: 93.35, 0.1: 96.25
                [140, 140, 140], #0.01: 87.50, 0.03: 91.98, 0.05: 93.44, 0.1: 96.44
                [150, 150, 150], #0.01: 87.23, 0.03: 92.25, 0.05: 93.45, 0.1: 95.98
                [160, 160, 160], #0.01: 88.30, 0.03: 92.08, 0.05: 93.22, 0.1: 96.09
                [180, 180, 180], #0.01: 87.63, 0.03: 92.80, 0.05: 94.02, 0.1: 96.00
                [200, 200, 200], #0.01: 87.45, 0.03: 92.40, 0.05: 94.22, 0.1: 96.15
                ]

for h in hidden_sizes:
    model = Deep_MLP(hidden_size_1=h[0], hidden_size_2=h[1], hidden_size_3=h[2], which_act=act)
    best_model, best_acc = model.train_model_best_ckpt(train_loader, test_loader, epochs=100, verbose=False)
    print(h[0], best_acc)