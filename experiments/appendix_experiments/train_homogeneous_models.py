"""
Training two MLP networks on the SAME MNIST data (no heterogeneous split).
For comparison with the heterogeneous split experiment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# =============================================================================
# CONFIGURATION - Adjust these parameters as needed
# =============================================================================

HIDDEN_LAYER_SIZES = [100, 100, 100]  # Three hidden layers
NUM_EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.5
SAVE_MODELS = True  # Save models after training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Activation function: 'relu' or 'gelu'
ACTIVATION = 'gelu'

# =============================================================================
# MODEL DEFINITION
# =============================================================================

def get_activation(name):
    """Return activation module based on name."""
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}. Use 'relu' or 'gelu'.")


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layer sizes and activation."""

    def __init__(self, input_size=784, hidden_sizes=None, output_size=10, activation='relu'):
        super(MLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [100, 100, 100]

        self.activation_name = activation
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def get_data_loaders():
    """Prepare MNIST data loaders - both models use full training set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    print(f"Data summary:")
    print(f"  Both models trained on: {len(train_dataset)} samples (full MNIST training set)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100. * correct / total


def train_model(model, train_loader, test_loader, model_name):
    """Train a model for NUM_EPOCHS."""
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {model_name}...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)
        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    return model


# =============================================================================
# MODEL SAVING/LOADING
# =============================================================================

def get_model_filename(model_name):
    """Generate filename based on model parameters."""
    hidden_str = "_".join(map(str, HIDDEN_LAYER_SIZES))
    return f"saved_models/{model_name}_homogeneous_{ACTIVATION}_h{hidden_str}_e{NUM_EPOCHS}.pt"


def save_model(model, model_name):
    """Save model to disk."""
    os.makedirs("saved_models", exist_ok=True)
    filename = get_model_filename(model_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_sizes': HIDDEN_LAYER_SIZES,
        'epochs': NUM_EPOCHS,
        'activation': ACTIVATION
    }, filename)
    print(f"  Saved {model_name} to {filename}")


def load_model(model_name):
    """Load model from disk if it exists with matching parameters."""
    filename = get_model_filename(model_name)
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=DEVICE)
        model = MLP(hidden_sizes=checkpoint['hidden_sizes'],
                    activation=checkpoint.get('activation', 'relu')).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded {model_name} from {filename}")
        return model
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("MLP Training on MNIST - Homogeneous Split (Same Data)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Hidden layers: {HIDDEN_LAYER_SIZES}")
    print(f"  Activation: {ACTIVATION}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"  Save models: {SAVE_MODELS}")

    # Load data
    train_loader, test_loader = get_data_loaders()

    # Try to load existing models or train new ones
    model_a = load_model("model_a") if SAVE_MODELS else None
    model_b = load_model("model_b") if SAVE_MODELS else None

    if model_a is None:
        model_a = MLP(hidden_sizes=HIDDEN_LAYER_SIZES, activation=ACTIVATION).to(DEVICE)
        model_a = train_model(model_a, train_loader, test_loader, "Model A")
        if SAVE_MODELS:
            save_model(model_a, "model_a")

    if model_b is None:
        model_b = MLP(hidden_sizes=HIDDEN_LAYER_SIZES, activation=ACTIVATION).to(DEVICE)
        model_b = train_model(model_b, train_loader, test_loader, "Model B")
        if SAVE_MODELS:
            save_model(model_b, "model_b")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Test Accuracies:")
    print("=" * 60)
    acc_a = evaluate(model_a, test_loader)
    acc_b = evaluate(model_b, test_loader)
    print(f"  Model A: {acc_a:.2f}%")
    print(f"  Model B: {acc_b:.2f}%")

    # Return models for further analysis
    return model_a, model_b


if __name__ == "__main__":
    model_a, model_b = main()
