from torch.utils.data import DataLoader, Subset, Dataset
import torch
from torchvision import datasets, transforms
import numpy as np


def load_mnist_subset(percentage=0.1, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def get_subset_indices(dataset, percentage):
        num_samples = len(dataset)
        indices = list(range(num_samples))
        split = int(np.floor(percentage * num_samples))
        np.random.shuffle(indices)
        return indices[:split]

    train_indices = get_subset_indices(full_train_dataset, percentage)
    test_indices = get_subset_indices(full_test_dataset, percentage)

    train_subset = Subset(full_train_dataset, train_indices)
    test_subset = Subset(full_test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_mnist(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_cifar10(batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def split_mnist_by_digit(batch_size=64, specific_digit=4, split_ratio=0.9):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def partition_and_split(dataset, specific_digit, split_ratio):
        specific_digit_indices = [i for i, label in enumerate(dataset.targets) if label == specific_digit]
        remaining_digits_indices = [i for i, label in enumerate(dataset.targets) if label != specific_digit]

        # Shuffle remaining digits indices and split them
        np.random.shuffle(remaining_digits_indices)
        split_point = int(len(remaining_digits_indices) * split_ratio)
        general_indices = remaining_digits_indices[:split_point]
        specialized_indices = remaining_digits_indices[split_point:] + specific_digit_indices

        # Create subsets
        general_subset = Subset(dataset, general_indices)
        specialized_subset = Subset(dataset, specialized_indices)

        # Create DataLoaders
        general_loader = DataLoader(general_subset, batch_size=batch_size, shuffle=True)
        specialized_loader = DataLoader(specialized_subset, batch_size=batch_size, shuffle=True)
        return general_loader, specialized_loader

    # Create loaders for training data
    train_general_loader, train_specialized_loader = partition_and_split(train_dataset, specific_digit, split_ratio)
    # Create loaders for test data (no splitting needed for test data)
    test_general_loader, test_specialized_loader = partition_and_split(test_dataset, specific_digit, split_ratio)

    # print("General train loader batches:", sum(1 for _ in train_general_loader))
    # print("Specialized train loader batches:", sum(1 for _ in train_specialized_loader))
    # print("General test loader batches:", sum(1 for _ in test_general_loader))
    # print("Specialized test loader batches:", sum(1 for _ in test_specialized_loader))

    return train_general_loader, test_general_loader, train_specialized_loader, test_specialized_loader



def split_mnist_by_digit_range(batch_size=64, split_ratio=0.9):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def get_indices_by_labels(dataset, labels):
        return [i for i, label in enumerate(dataset.targets) if label in labels]

    def create_biased_subset(dataset, primary_digits, secondary_digits, primary_ratio):
        primary_indices = get_indices_by_labels(dataset, primary_digits)
        secondary_indices = get_indices_by_labels(dataset, secondary_digits)

        # Shuffle indices
        np.random.shuffle(primary_indices)
        np.random.shuffle(secondary_indices)

        # Split
        num_primary = int(len(primary_indices) * primary_ratio)
        num_secondary = int(len(secondary_indices) * (1 - primary_ratio))

        selected_indices = primary_indices[:num_primary] + secondary_indices[:num_secondary]
        return Subset(dataset, selected_indices)

    def create_loaders(dataset):
        group_a_subset = create_biased_subset(dataset, primary_digits=range(0, 4), secondary_digits=range(5, 9), primary_ratio=split_ratio)
        group_b_subset = create_biased_subset(dataset, primary_digits=range(5, 9), secondary_digits=range(0, 4), primary_ratio=split_ratio)

        loader_a = DataLoader(group_a_subset, batch_size=batch_size, shuffle=True)
        loader_b = DataLoader(group_b_subset, batch_size=batch_size, shuffle=True)
        return loader_a, loader_b

    train_loader_a, train_loader_b = create_loaders(train_dataset)
    test_loader_a, test_loader_b = create_loaders(test_dataset)

    return train_loader_a, test_loader_a, train_loader_b, test_loader_b




class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size, max_int):
        """
        A dataset that generates random samples with one-dimensional integer outputs.

        Args:
            num_samples (int): Number of samples in the dataset.
            input_size (int): Size of the input vector.
            max_int (int): Maximum integer value for the output (0 to max_int, inclusive).
        """
        self.num_samples = num_samples
        self.input_size = input_size
        self.max_int = max_int

        # Generate random data
        self.inputs = np.random.rand(num_samples, input_size).astype(np.float32)
        self.outputs = np.random.randint(0, max_int + 1, size=(num_samples,), dtype=np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.outputs[idx])


def load_random_data(num_samples=10000, input_size=10, max_int=10, batch_size=64):
    """
    Create a DataLoader for random data.

    Args:
        num_samples (int): Number of samples to generate.
        input_size (int): Size of the input vector.
        max_int (int): Maximum integer value for the output (0 to max_int, inclusive).
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: A DataLoader instance for random data.
    """
    dataset = RandomDataset(num_samples, input_size, max_int)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

