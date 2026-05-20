import sys
import torch
sys.path.append("../")
from src.CNN import VGG11
from src import data_loader

train_loader, test_loader = data_loader.load_cifar10()
model_size = [64, 128, 256, 256, 512, 512, 512, 512]

mults = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

hidden_sizes = [[int(layer_size * mult) for layer_size in model_size] for mult in mults]

for h, mult in zip(hidden_sizes, mults):
    model = VGG11(manual_chanel_sizes=h, neurons_classifier=h[-1])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    best_model, best_acc = model.train_model_best_ckpt(train_loader, test_loader, epochs=200, verbose=False)
    print(mult, best_acc)