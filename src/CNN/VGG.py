import torch.nn as nn
import torch
import numpy as np
from src.base_model import BaseModel
from torchvision.models import vgg11
from collections import OrderedDict


class VGG11(BaseModel):
    def __init__(self, num_classes=10, manual_chanel_sizes=None, neurons_classifier=512, which_act=0):
        super().__init__()
        # The original VGG11 has a features and a classifier section.
        # We will manually recreate the 'features' part here with named layers.
        base = vgg11(weights=None)
        if which_act == 0:
            act_func1 = nn.ReLU(inplace=True)
            act_func2 = nn.ReLU(inplace=True)
            act_func3 = nn.ReLU(inplace=True)
        elif which_act == 1:
            act_func1 = nn.LeakyReLU(inplace=True)
            act_func2 = nn.LeakyReLU(inplace=True)
            act_func3 = nn.LeakyReLU(inplace=True)
        elif which_act == 2:
            act_func1 = nn.GELU()
            act_func2 = nn.GELU()
            act_func3 = nn.GELU()
        else:
            raise NotImplementedError
        indices = [0, 3, 6, 8, 11, 13, 16, 18]
        input_channels = 3
        channel_sizes = []
        for index in indices:
            channel_sizes.append(base.features[index].out_channels)
        if manual_chanel_sizes is not None:
            channel_sizes = manual_chanel_sizes
        print(channel_sizes)

        # Manually define the layers of the VGG11 model.
        # This replaces the dynamic loop from the original code.
        self.conv1 = nn.Conv2d(
            input_channels,
            channel_sizes[0],
            kernel_size=base.features[0].kernel_size,
            stride=base.features[0].stride,
            padding=base.features[0].padding,
            bias=False
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            channel_sizes[0],
            channel_sizes[1],
            kernel_size=base.features[3].kernel_size,
            stride=base.features[3].stride,
            padding=base.features[3].padding,
            bias=False
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            channel_sizes[1],
            channel_sizes[2],
            kernel_size=base.features[6].kernel_size,
            stride=base.features[6].stride,
            padding=base.features[6].padding,
            bias=False
        )
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(
            channel_sizes[2],
            channel_sizes[3],
            kernel_size=base.features[8].kernel_size,
            stride=base.features[8].stride,
            padding=base.features[8].padding,
            bias=False
        )
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            channel_sizes[3],
            channel_sizes[4],
            kernel_size=base.features[11].kernel_size,
            stride=base.features[11].stride,
            padding=base.features[11].padding,
            bias=False
        )
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(
            channel_sizes[4],
            channel_sizes[5],
            kernel_size=base.features[13].kernel_size,
            stride=base.features[13].stride,
            padding=base.features[13].padding,
            bias=False
        )
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(
            channel_sizes[5],
            channel_sizes[6],
            kernel_size=base.features[16].kernel_size,
            stride=base.features[16].stride,
            padding=base.features[16].padding,
            bias=False
        )
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(
            channel_sizes[6],
            channel_sizes[7],
            kernel_size=base.features[18].kernel_size,
            stride=base.features[18].stride,
            padding=base.features[18].padding,
            bias=False
        )
        self.relu8 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(neurons_classifier, num_classes, bias=False)

    def forward(self, x):
        # The forward pass must now explicitly call each named layer in order.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool4(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.maxpool5(x)

        # Pass through the classifier section
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def load_reference_model(self, file_path, cpu=False):
        device = torch.device("cpu") if cpu else self.device
        checkpoint = torch.load(file_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        # This dictionary maps the old numeric keys to the new named keys.
        key_mapping = {
            "features.0.weight": "conv1.weight",
            "features.3.weight": "conv2.weight",
            "features.6.weight": "conv3.weight",
            "features.8.weight": "conv4.weight",
            "features.11.weight": "conv5.weight",
            "features.13.weight": "conv6.weight",
            "features.16.weight": "conv7.weight",
            "features.18.weight": "conv8.weight",
            "classifier.6.weight": "linear.weight",
            "classifier.weight": "linear.weight",
        }

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in key_mapping:
                new_key = key_mapping[k]
                new_state_dict[new_key] = v
            else:
                # Handle cases where keys don't need remapping, like biases
                # if they were ever included.
                new_state_dict[k] = v

        # Load the remapped state dictionary
        self.load_state_dict(new_state_dict, strict=True)
        self.to(device)




def _reconstruct_network(net, layers, pi_list, smaller_sizes):
    weight_list = []

    # Input dim of the network
    input_dim = net.get_incoming_weights(layers[0], numpy=True).shape[1]
    k_back_prev = np.eye(input_dim)

    for l, layer_name in enumerate(layers):
        w_old = net.get_incoming_weights(layer_name, numpy=True)
        pi = pi_list[l]

        w_mu = np.sum(pi, axis=1)
        w_nu = np.sum(pi, axis=0)

        k_for = np.transpose(np.divide(pi, w_mu[:, None], out=np.zeros_like(pi), where=w_mu[:, None] != 0))
        k_back = np.divide(pi, w_nu[None, :], out=np.zeros_like(pi), where=w_nu[None, :] != 0)

        if len(w_old.shape) == 2:
            w_new = k_for @ w_old @ k_back_prev
        else:
            tmp = np.einsum('bchw,cd->bdhw', w_old, k_back_prev)
            w_new = np.tensordot(k_for, tmp, axes=([1], [0]))

        weight_list.append(w_new)
        k_back_prev = k_back

        # --- Build New Net ---
    out_net = VGG11(manual_chanel_sizes=smaller_sizes, neurons_classifier=smaller_sizes[-1])
    for l, layer_name in enumerate(layers):
        out_net.set_incoming_weights(layer_name, weight_list[l], numpy=True)

    # --- Final Layer Safety Check ---
    # If the network has a final linear layer ('linear') that was NOT in 'layers',
    # we must adapt its input. If it WAS in 'layers', it's already done.

    last_layer_name = 'linear'

    if last_layer_name not in layers:
        print('last layer was not pruned')
        w_last = net.get_incoming_weights(last_layer_name, numpy=True)
        # Handle Conv->Linear or Linear->Linear transition
        if w_last.shape[1] != k_back_prev.shape[0]:
            # Conv -> Linear
            n_channels_old = k_back_prev.shape[0]
            flattened_dim = w_last.shape[1]
            spatial_area = flattened_dim // n_channels_old

            w_last_reshaped = w_last.reshape(w_last.shape[0], n_channels_old, spatial_area)
            w_last_contracted = np.einsum('ocs,cn->ons', w_last_reshaped, k_back_prev)
            w_last_new = w_last_contracted.reshape(w_last.shape[0], -1)
        else:
            # Linear -> Linear
            w_last_new = np.einsum('oi,id->od', w_last, k_back_prev)

        out_net.set_incoming_weights(last_layer_name, w_last_new, numpy=True)

    return out_net