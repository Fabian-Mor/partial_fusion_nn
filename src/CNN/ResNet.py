import torch
import torch.nn as nn
import numpy as np
from src.base_model import BaseModel


class ResNet18(BaseModel):
    """
    ResNet18 for CIFAR10 (32x32 input), following the same conventions as VGG11.

    All layers are explicitly named so that get_layer_names(), get_layer_by_name(),
    get_incoming_weights(), etc. work correctly via BaseModel.

    Architecture:
        conv1 -> bn1 -> relu0
        layer1: block0 (conv1, bn1, conv2, bn2) + block1 (conv1, bn1, conv2, bn2)
        layer2: block0 (conv1, bn1, conv2, bn2, downsample.0, downsample.1) + block1
        layer3: block0 (conv1, bn1, conv2, bn2, downsample.0, downsample.1) + block1
        layer4: block0 (conv1, bn1, conv2, bn2, downsample.0, downsample.1) + block1
        avgpool -> fc

    Naming convention for layers:
        layerX_blockY_convZ   (Conv2d, has weights)
        layerX_blockY_bnZ     (BatchNorm2d, has weights but is a norm layer)
        layerX_blockY_ds_conv (downsample conv, has weights, is a residual layer)
        layerX_blockY_ds_bn   (downsample bn, has weights, is a norm layer)
    """

    def __init__(self, num_classes=10, manual_channel_sizes=None):
        super().__init__()
        self.input_size = None  # CNN, no flattening at input

        if manual_channel_sizes is None:
            channels = [64, 64, 128, 256, 512]
        else:
            channels = manual_channel_sizes

        # --- Stem ---
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu0 = nn.ReLU(inplace=True)

        # --- Layer 1 (no downsampling) ---
        # Block 0
        self.layer1_block0_conv1 = nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=1, bias=False)
        self.layer1_block0_bn1 = nn.BatchNorm2d(channels[1])
        self.layer1_block0_conv2 = nn.Conv2d(channels[1], channels[1], 3, stride=1, padding=1, bias=False)
        self.layer1_block0_bn2 = nn.BatchNorm2d(channels[1])
        # Block 1
        self.layer1_block1_conv1 = nn.Conv2d(channels[1], channels[1], 3, stride=1, padding=1, bias=False)
        self.layer1_block1_bn1 = nn.BatchNorm2d(channels[1])
        self.layer1_block1_conv2 = nn.Conv2d(channels[1], channels[1], 3, stride=1, padding=1, bias=False)
        self.layer1_block1_bn2 = nn.BatchNorm2d(channels[1])

        # --- Layer 2 (downsample) ---
        # Block 0
        self.layer2_block0_conv1 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.layer2_block0_bn1 = nn.BatchNorm2d(channels[2])
        self.layer2_block0_conv2 = nn.Conv2d(channels[2], channels[2], 3, stride=1, padding=1, bias=False)
        self.layer2_block0_bn2 = nn.BatchNorm2d(channels[2])
        self.layer2_block0_ds_conv = nn.Conv2d(channels[1], channels[2], 1, stride=2, bias=False)
        self.layer2_block0_ds_bn = nn.BatchNorm2d(channels[2])
        # Block 1
        self.layer2_block1_conv1 = nn.Conv2d(channels[2], channels[2], 3, stride=1, padding=1, bias=False)
        self.layer2_block1_bn1 = nn.BatchNorm2d(channels[2])
        self.layer2_block1_conv2 = nn.Conv2d(channels[2], channels[2], 3, stride=1, padding=1, bias=False)
        self.layer2_block1_bn2 = nn.BatchNorm2d(channels[2])

        # --- Layer 3 (downsample) ---
        self.layer3_block0_conv1 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.layer3_block0_bn1 = nn.BatchNorm2d(channels[3])
        self.layer3_block0_conv2 = nn.Conv2d(channels[3], channels[3], 3, stride=1, padding=1, bias=False)
        self.layer3_block0_bn2 = nn.BatchNorm2d(channels[3])
        self.layer3_block0_ds_conv = nn.Conv2d(channels[2], channels[3], 1, stride=2, bias=False)
        self.layer3_block0_ds_bn = nn.BatchNorm2d(channels[3])
        # Block 1
        self.layer3_block1_conv1 = nn.Conv2d(channels[3], channels[3], 3, stride=1, padding=1, bias=False)
        self.layer3_block1_bn1 = nn.BatchNorm2d(channels[3])
        self.layer3_block1_conv2 = nn.Conv2d(channels[3], channels[3], 3, stride=1, padding=1, bias=False)
        self.layer3_block1_bn2 = nn.BatchNorm2d(channels[3])

        # --- Layer 4 (downsample) ---
        self.layer4_block0_conv1 = nn.Conv2d(channels[3], channels[4], 3, stride=2, padding=1, bias=False)
        self.layer4_block0_bn1 = nn.BatchNorm2d(channels[4])
        self.layer4_block0_conv2 = nn.Conv2d(channels[4], channels[4], 3, stride=1, padding=1, bias=False)
        self.layer4_block0_bn2 = nn.BatchNorm2d(channels[4])
        self.layer4_block0_ds_conv = nn.Conv2d(channels[3], channels[4], 1, stride=2, bias=False)
        self.layer4_block0_ds_bn = nn.BatchNorm2d(channels[4])
        # Block 1
        self.layer4_block1_conv1 = nn.Conv2d(channels[4], channels[4], 3, stride=1, padding=1, bias=False)
        self.layer4_block1_bn1 = nn.BatchNorm2d(channels[4])
        self.layer4_block1_conv2 = nn.Conv2d(channels[4], channels[4], 3, stride=1, padding=1, bias=False)
        self.layer4_block1_bn2 = nn.BatchNorm2d(channels[4])

        # --- Head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[4], num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _basic_block(self, x, prefix, has_downsample=False):
        """Run a basic residual block given the layer name prefix."""
        identity = x

        out = getattr(self, f'{prefix}_conv1')(x)
        out = getattr(self, f'{prefix}_bn1')(out)
        out = torch.relu(out)
        out = getattr(self, f'{prefix}_conv2')(out)
        out = getattr(self, f'{prefix}_bn2')(out)

        if has_downsample:
            identity = getattr(self, f'{prefix}_ds_conv')(x)
            identity = getattr(self, f'{prefix}_ds_bn')(identity)

        out = out + identity
        out = torch.relu(out)
        return out

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu0(x)

        # Layer 1 (no downsampling, channels[0] -> channels[1], same if equal)
        x = self._basic_block(x, 'layer1_block0', has_downsample=False)
        x = self._basic_block(x, 'layer1_block1', has_downsample=False)

        # Layer 2
        x = self._basic_block(x, 'layer2_block0', has_downsample=True)
        x = self._basic_block(x, 'layer2_block1', has_downsample=False)

        # Layer 3
        x = self._basic_block(x, 'layer3_block0', has_downsample=True)
        x = self._basic_block(x, 'layer3_block1', has_downsample=False)

        # Layer 4
        x = self._basic_block(x, 'layer4_block0', has_downsample=True)
        x = self._basic_block(x, 'layer4_block1', has_downsample=False)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_residual_layers(self):
        """
        Returns information about residual (skip) connections.

        Returns:
            residual_conv_layers: list of layer names for downsample convs
                (these are excluded from the main fusion loop)
            residual_norm_layers: list of layer names for downsample bns
            residual_map: dict mapping each ds_conv to (input_main_conv, output_main_conv)
                i.e. which main-branch convs the skip connection spans
            norm_map: dict mapping each main-branch conv to its following bn
            identity_residual_map: dict mapping virtual skip names to (first_conv, last_conv)
                for blocks WITHOUT downsample — used by partial fusion to create
                explicit skip connections when channel dimensions expand
        """
        residual_conv_layers = [
            'layer2_block0_ds_conv',
            'layer3_block0_ds_conv',
            'layer4_block0_ds_conv',
        ]
        residual_norm_layers = [
            'layer2_block0_ds_bn',
            'layer3_block0_ds_bn',
            'layer4_block0_ds_bn',
        ]
        # Maps downsample conv -> (first conv in block, last conv in block)
        # This tells the fusion: the skip conv has the same input space as the
        # first conv and the same output space as the last conv in the block.
        residual_map = {
            'layer2_block0_ds_conv': ('layer2_block0_conv1', 'layer2_block0_conv2'),
            'layer3_block0_ds_conv': ('layer3_block0_conv1', 'layer3_block0_conv2'),
            'layer4_block0_ds_conv': ('layer4_block0_conv1', 'layer4_block0_conv2'),
        }
        # Identity residual connections (blocks without downsample).
        # In the original model these are plain identity shortcuts, but after
        # partial fusion the channel count / ordering may change, requiring an
        # explicit 1x1 projection in the fused model.
        identity_residual_map = {
            'layer1_block0_skip': ('layer1_block0_conv1', 'layer1_block0_conv2'),
            'layer1_block1_skip': ('layer1_block1_conv1', 'layer1_block1_conv2'),
            'layer2_block1_skip': ('layer2_block1_conv1', 'layer2_block1_conv2'),
            'layer3_block1_skip': ('layer3_block1_conv1', 'layer3_block1_conv2'),
            'layer4_block1_skip': ('layer4_block1_conv1', 'layer4_block1_conv2'),
        }
        # Maps each conv to its following batch norm (for recalibration)
        norm_map = {}
        for name, _ in self.named_modules():
            if name.endswith('_conv1') or name.endswith('_conv2'):
                bn_name = name.replace('_conv', '_bn')
                if hasattr(self, bn_name):
                    norm_map[name] = bn_name
            elif name == 'conv1':
                norm_map['conv1'] = 'bn1'
            elif name.endswith('_ds_conv'):
                bn_name = name.replace('_ds_conv', '_ds_bn')
                if hasattr(self, bn_name):
                    norm_map[name] = bn_name

        return residual_conv_layers, residual_norm_layers, residual_map, norm_map, identity_residual_map

    def get_norm_layer_names(self):
        """Return names of all BatchNorm layers."""
        return [name for name, m in self.named_modules() if isinstance(m, nn.BatchNorm2d)]

    def load_from_torchvision(self, torchvision_resnet):
        """Load weights from a torchvision ResNet18 model."""
        tv = torchvision_resnet
        mapping = {
            'conv1': tv.conv1, 'bn1': tv.bn1,
        }
        for li in range(1, 5):
            layer = getattr(tv, f'layer{li}')
            for bi in range(2):
                block = layer[bi]
                prefix = f'layer{li}_block{bi}'
                mapping[f'{prefix}_conv1'] = block.conv1
                mapping[f'{prefix}_bn1'] = block.bn1
                mapping[f'{prefix}_conv2'] = block.conv2
                mapping[f'{prefix}_bn2'] = block.bn2
                if hasattr(block, 'downsample') and block.downsample is not None:
                    mapping[f'{prefix}_ds_conv'] = block.downsample[0]
                    mapping[f'{prefix}_ds_bn'] = block.downsample[1]

        for name, src_module in mapping.items():
            tgt_module = self.get_layer_by_name(name)
            if tgt_module is not None:
                tgt_module.load_state_dict(src_module.state_dict())
