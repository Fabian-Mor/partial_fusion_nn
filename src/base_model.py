import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.activation_classes = {
            nn.ReLU: (torch.relu, lambda x: np.maximum(x, 0)),
            nn.Sigmoid: (torch.sigmoid, lambda x: 1 / (1 + np.exp(-x))),
            nn.Tanh: (torch.tanh, np.tanh),
            nn.Softmax: (
                lambda x: torch.nn.functional.softmax(x, dim=-1),
                lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True),
            ),
            nn.LeakyReLU: (
                lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01),
                lambda x: np.maximum(x, 0.01 * x),
            ),
        }

    def forward(self, x):
        """
        Forward pass, assume the activation is applied separately after each layer
        """
        raise NotImplementedError

    def get_layer_names(self):
        layer_names = []
        for name, layer in self.named_modules():
            if name != '':
                layer_names.append(name)
        return layer_names

    def get_layer_names_with_weights(self):
        """
        Get names of all layers that have incoming weights.
        """
        layer_names = []
        for name, layer in self.named_modules():
            # Check if the layer has a 'weight' attribute
            if name != '' and hasattr(layer, 'weight'):
                layer_names.append(name)
        return layer_names

    def get_layer_by_name(self, layer_name):
        """
        Given a layer name, return the actual layer.

        Args:
            layer_name (str): The name of the layer.

        Returns:
            nn.Module: The layer corresponding to the given name, or None if not found.
        """
        for name, layer in self.named_modules():
            if name == layer_name:
                return layer
        return None

    def get_activations(self, layer_name, x, numpy=False):
        """
        Access activations of a specific layer given input x
        """
        if isinstance(x, (np.ndarray,)):
            x = torch.from_numpy(x)
        self.to(self.device)
        x = x.to(self.device)
        activations = {}
        def hook_fn(module, input, output):
            activations[layer_name] = output
        layer = dict(self.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            self(x)
        hook.remove()
        if activations.get(layer_name, None) is None:
            return None
        if numpy:
            return activations.get(layer_name, None).detach().cpu().numpy()
        return activations.get(layer_name, None)

    def get_all_activations(self, x, numpy=False, include_relu=True, include_norm=True):
        layers = self.get_layer_names()
        activations = {}
        for layer_name in layers:
            if 'dropout' in layer_name:
                continue
            if not include_norm and ('norm' in layer_name):
                continue
            if not include_relu and ('relu' in layer_name or 'ReLU' in layer_name):
                continue
            try:
                act = self.get_activations(layer_name, x, numpy=numpy).T
            except AttributeError:
                continue
            if act is not None:
                activations[layer_name] = act
        return activations

    def get_incoming_weights(self, layer_name, numpy=False):
        """
        Access the incoming weights of a specific layer.
        """
        weights = {}
        layer = dict(self.named_modules())[layer_name]
        if hasattr(layer, 'weight'):
            weights[layer_name] = layer.weight
        if numpy:
            return weights.get(layer_name, None).detach().cpu().numpy()
        return weights.get(layer_name, None)

    def set_incoming_weights(self, layer_name, weights, numpy=False):
        """
        Set the incoming weights of a specific layer.

        Args:
            layer_name (str): The name of the layer whose weights will be updated.
            weights (Tensor or numpy.ndarray): The new weights for the layer.
            numpy (bool): Whether the provided weights are in NumPy format.
        """
        layer = dict(self.named_modules()).get(layer_name, None)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

        if not hasattr(layer, 'weight'):
            raise ValueError(f"Layer '{layer_name}' does not have a weight attribute.")

        if numpy:
            if not isinstance(weights, (np.ndarray,)):
                raise TypeError("Weights must be a numpy.ndarray if numpy=True.")
            weights = torch.from_numpy(weights)

        if weights.shape != layer.weight.shape:
            raise ValueError(f"Shape mismatch: Expected {layer.weight.shape}, got {weights.shape}.")

        with torch.no_grad():
            layer.weight.copy_(weights)


    def get_next_activation(self, layer_name, numpy=False):
        found_layer = False
        for name, layer in self.named_modules():
            if found_layer:
                # Check if the layer is a known activation
                for act_class, (torch_func, numpy_func) in self.activation_classes.items():
                    if isinstance(layer, act_class):
                        # Return the appropriate function based on the `numpy` flag
                        return numpy_func if numpy else torch_func
                return None  # Next layer is not a known activation

            elif name == layer_name:
                found_layer = True
        return None


    def get_total_weights(self):
        total = 0
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                total += param.numel()
        return total


    def get_skip_connections(self):
        return {}

    def get_residual_layers(self):
        return [], [], {}, {}

    def save_model(self, file_path):
        """
        Save model to a file
        """
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path, cpu=False):
        """
        Load model from a file
        """
        self.load_state_dict(torch.load(file_path, weights_only=True, map_location=torch.device('cpu')))
        self.to(self.device)

    def train_model(self, train_loader, epochs=5, lr=0.001, optimizer='', loss_func=nn.CrossEntropyLoss(), verbose=True):
        self.to(self.device)
        self.train()  # Set model to training mode
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr) # set to Adam by default
        criterion = loss_func
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            if self.scheduler is not None:
                self.scheduler.step()
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    def test_model(self, test_loader, verbose=True, criterion=None):
        if criterion is None:
            criterion = self.accuracy
        self.to(self.device)
        self.eval()  # Set model to evaluation mode
        acc = 0
        total_samples = 0
        with torch.no_grad():  # Disable gradient tracking for inference
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                batch_loss = criterion(outputs, labels)
                if isinstance(batch_loss, torch.Tensor):
                    batch_loss = batch_loss.item()
                acc += batch_loss * labels.size(0)  # Scale loss by batch size
                total_samples += labels.size(0)  # Accumulate the total number of samples

        accuracy = acc / total_samples  # Average loss per sample
        if verbose:
            print(f"Test metric: {accuracy:.2f}")
        return accuracy

    def accuracy(self, outputs, labels):
        predicted = outputs.argmax(dim=1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct / total * 100

    def get_lambda(self, l):
        return 1

    def multiply_weights(self, factors):
        L = len(self.get_layer_names_with_weights())
        for idx, name in enumerate(self.get_layer_names_with_weights()):
            weights = self.get_incoming_weights(name)
            self.set_incoming_weights(name, weights * factors[idx % L])



    def train_model_best_ckpt(self, train_loader, test_loader, epochs=5, lr=0.001, optimizer='', loss_func=nn.CrossEntropyLoss(), verbose=True):
        self.to(self.device)
        self.train()  # Set model to training mode
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr) # set to Adam by default
        criterion = loss_func
        best_model = copy.deepcopy(self)
        best_acc = 0
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if self.scheduler is not None:
                self.scheduler.step()
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
            acc = self.test_model(test_loader, verbose=False)
            if acc > best_acc:
                if verbose:
                    print(f'new best model with accuracy {acc}')
                best_model = copy.deepcopy(self)
                best_acc = acc
        return best_model, best_acc