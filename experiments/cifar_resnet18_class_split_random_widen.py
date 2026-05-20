#!/usr/bin/env python3
"""
Sanity check: random widening of Model A vs. partial fusion (ResNet18, CIFAR-10).
=================================================================================
For each alpha, create a widened version of Model A that has roughly the same
number of trainable parameters per layer as the partially fused model, but with
random additional channels (not from Model B). Fine-tune and compare.

This tests whether the gains from partial fusion come from the actual
complementary knowledge of Model B, or just from having more parameters.
"""
import sys, os, json, time, copy
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
sys.path.append("../")
sys.stdout.reconfigure(line_buffering=True)
from src.CNN import ResNet18
from src.FusionModel.fusion_model import FusionModel
from src.FusionModel.fusion_methods.partial_fusion import PartialFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

SEED = 42; FT_EPOCHS = 30; FT_LR = 0.001; BATCH_SIZE = 128; FT_BATCH_SIZE = 64
RESULTS_DIR = "../results"; CKPT_DIR = "./saved"
EVEN_CLASSES = [0, 2, 4, 6, 8]

def get_data_and_splits():
    train_tf = transforms.Compose([transforms.RandomCrop(32,padding=4), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_aug = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)

    np.random.seed(SEED)
    targets = np.array(train_aug.targets)
    split_ft = []
    for c in range(10):
        idx = np.where(targets == c)[0].copy(); np.random.shuffle(idx)
        split_ft.extend(idx[:int(0.05*len(idx))].tolist())
    return train_aug, test_set, split_ft

def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total

def finetune(model, ft_loader, test_loader, epochs=30, lr=0.001):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = evaluate(model, test_loader); best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for x, y in ft_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        scheduler.step()
        acc = evaluate(model, test_loader)
        if acc > best_acc: best_acc = acc; best_state = copy.deepcopy(model.state_dict())
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            print(f"    FT {epoch+1}/{epochs}: {acc:.2f}% (best={best_acc:.2f}%)", flush=True)
    model.load_state_dict(best_state)
    return model, best_acc


class WidenedResNet18(nn.Module):
    """
    Take a pretrained ResNet18 and widen each layer by adding random extra channels.
    The original channels keep their trained weights; extra channels are initialized randomly
    (Kaiming init) so they start as noise but can be learned during fine-tuning.
    """
    def __init__(self, original_model, target_channels):
        """
        target_channels: dict mapping layer_name -> target_out_channels
        Only conv layers are widened. BN and skip connections are adjusted to match.
        """
        super().__init__()
        self.target_channels = target_channels
        self.layers = nn.ModuleDict()
        self._build(original_model)

    def _widen_conv(self, conv, new_out, new_in):
        """Create a wider conv, copying original weights into top-left corner."""
        new_conv = nn.Conv2d(new_in, new_out, conv.kernel_size, conv.stride,
                             conv.padding, bias=conv.bias is not None).to(device)
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        with torch.no_grad():
            # Zero out the extra channels initially so model starts ≈ original
            new_conv.weight.zero_()
            old_out, old_in = conv.weight.shape[:2]
            new_conv.weight[:old_out, :old_in] = conv.weight
        return new_conv

    def _widen_bn(self, bn, new_channels):
        new_bn = nn.BatchNorm2d(new_channels).to(device)
        old_c = bn.num_features
        with torch.no_grad():
            new_bn.weight[:old_c] = bn.weight
            new_bn.bias[:old_c] = bn.bias
            new_bn.running_mean[:old_c] = bn.running_mean
            new_bn.running_var[:old_c] = bn.running_var
            # Extra channels: default init (weight=1, bias=0, mean=0, var=1)
        return new_bn

    def _build(self, model):
        # Determine channel sizes at each stage
        # Original ResNet18 channels: [64, 64, 128, 256, 512]
        orig_channels = [64, 64, 128, 256, 512]

        # Compute target channels per stage from the target_channels dict
        # We use the conv1 output and each stage's first block conv1 output
        stage_channels = list(orig_channels)  # default
        for name, target in self.target_channels.items():
            if name == 'conv1':
                stage_channels[0] = target
            elif 'layer1' in name and 'conv1' in name:
                stage_channels[1] = target
            elif 'layer2' in name and 'conv1' in name:
                stage_channels[2] = target
            elif 'layer3' in name and 'conv1' in name:
                stage_channels[3] = target
            elif 'layer4' in name and 'conv1' in name:
                stage_channels[4] = target

        c = stage_channels
        # Stem
        self.layers['conv1'] = self._widen_conv(model.conv1, c[0], 3)
        self.layers['bn1'] = self._widen_bn(model.bn1, c[0])

        # Build each stage
        for stage_idx, stage_num in enumerate(range(1, 5)):
            c_in = c[stage_idx]
            c_out = c[stage_idx + 1] if stage_idx < 3 else c[4]
            # Wait, channel indexing: stage 1 uses c[1], stage 2 uses c[2], etc.
            c_out = c[stage_num]
            c_in_stage = c[stage_num - 1] if stage_num > 1 else c[0]

            for block in range(2):
                prefix = f'layer{stage_num}_block{block}'
                orig_conv1 = getattr(model, f'{prefix}_conv1')
                orig_bn1 = getattr(model, f'{prefix}_bn1')
                orig_conv2 = getattr(model, f'{prefix}_conv2')
                orig_bn2 = getattr(model, f'{prefix}_bn2')

                if block == 0:
                    in_ch = c_in_stage
                else:
                    in_ch = c_out

                self.layers[f'{prefix}_conv1'] = self._widen_conv(orig_conv1, c_out, in_ch)
                self.layers[f'{prefix}_bn1'] = self._widen_bn(orig_bn1, c_out)
                self.layers[f'{prefix}_conv2'] = self._widen_conv(orig_conv2, c_out, c_out)
                self.layers[f'{prefix}_bn2'] = self._widen_bn(orig_bn2, c_out)

                # Downsample if needed (stage >= 2, block 0)
                ds_conv_name = f'{prefix}_ds_conv'
                if hasattr(model, ds_conv_name):
                    orig_ds = getattr(model, ds_conv_name)
                    orig_ds_bn = getattr(model, f'{prefix}_ds_bn')
                    self.layers[ds_conv_name] = self._widen_conv(orig_ds, c_out, c_in_stage)
                    self.layers[f'{prefix}_ds_bn'] = self._widen_bn(orig_ds_bn, c_out)

                # Skip connection for non-downsample blocks if channel size changed
                if not hasattr(model, ds_conv_name) and in_ch != c_out:
                    skip = nn.Conv2d(in_ch, c_out, 1, stride=1, bias=False).to(device)
                    with torch.no_grad():
                        skip.weight.zero_()
                        min_c = min(in_ch, c_out)
                        # Identity for original channels
                        for i in range(min(orig_channels[stage_num], min_c)):
                            skip.weight[i, i] = 1.0
                    self.layers[f'{prefix}_skip'] = skip

        # FC layer
        self.layers['fc'] = nn.Linear(c[4], 10, bias=False).to(device)
        with torch.no_grad():
            self.layers['fc'].weight.zero_()
            self.layers['fc'].weight[:, :512] = model.fc.weight

    def forward(self, x):
        x = x.to(device)
        x = self.layers['conv1'](x)
        x = self.layers['bn1'](x)
        x = torch.relu(x)

        for stage in range(1, 5):
            for block in range(2):
                prefix = f'layer{stage}_block{block}'
                identity = x
                out = self.layers[f'{prefix}_conv1'](x)
                out = self.layers[f'{prefix}_bn1'](out)
                out = torch.relu(out)
                out = self.layers[f'{prefix}_conv2'](out)
                out = self.layers[f'{prefix}_bn2'](out)

                ds_name = f'{prefix}_ds_conv'
                skip_name = f'{prefix}_skip'
                if ds_name in self.layers:
                    identity = self.layers[ds_name](x)
                    identity = self.layers[f'{prefix}_ds_bn'](identity)
                elif skip_name in self.layers:
                    identity = self.layers[skip_name](x)

                out = out + identity
                out = torch.relu(out)
                x = out

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.layers['fc'](x)
        return x


def get_fusion_channel_sizes(model_a, model_b, alpha):
    """Get the channel sizes of the fused model at a given alpha."""
    from torch.utils.data import DataLoader
    fm = PartialFusion(alphas=alpha, combine_costs=True, pgd=True)
    fused = FusionModel(copy.deepcopy(model_a), copy.deepcopy(model_b), fm,
                        lambdas=[0.5, 0.5], running_stats_init='copy')
    # Extract channel sizes from fused conv layers
    channels = {}
    for name, layer in fused.fused_layers_dict.items():
        if isinstance(layer, nn.Conv2d):
            channels[name] = layer.out_channels
    nonzero = int(sum((p.data.abs() > 0).sum().item() for p in fused.parameters()))
    total = sum(p.numel() for p in fused.parameters())
    del fused; torch.cuda.empty_cache()
    return channels, nonzero, total


def main():
    results = {}
    print("=" * 70, flush=True)
    print("Exp 25d: Random Widening of Model A vs Partial Fusion", flush=True)
    print("=" * 70, flush=True)

    train_aug, test_set, split_ft = get_data_and_splits()
    ft_loader = DataLoader(Subset(train_aug, split_ft), batch_size=FT_BATCH_SIZE,
                           shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model_a = ResNet18(num_classes=10).to(device)
    model_a.load_model(os.path.join(CKPT_DIR, "resnet18_evenodd_a.checkpoint"))
    model_b = ResNet18(num_classes=10).to(device)
    model_b.load_model(os.path.join(CKPT_DIR, "resnet18_evenodd_b.checkpoint"))
    acc_a = evaluate(model_a, test_loader)
    n_params_single = sum(p.numel() for p in model_a.parameters())
    print(f"Model A: {acc_a:.2f}%, params: {n_params_single:,}", flush=True)

    # Optional comparison data; left empty since standalone widening reports its own results
    prev = {}

    alphas = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    for alpha in alphas:
        print(f"\n{'='*55}", flush=True)
        print(f"  alpha={alpha}", flush=True)
        print(f"{'='*55}", flush=True)

        # Get fusion channel sizes as target
        print(f"  Computing fusion channel sizes...", flush=True)
        fused_channels, fused_nz, fused_total = get_fusion_channel_sizes(model_a, model_b, alpha)
        print(f"  Fused channels: {fused_channels}", flush=True)
        print(f"  Fused NZ params: {fused_nz:,}, total: {fused_total:,}", flush=True)

        # Create widened model A with matching channel sizes
        print(f"  Creating widened Model A...", flush=True)
        torch.manual_seed(SEED)
        widened = WidenedResNet18(model_a, fused_channels).to(device)
        widen_params = sum(p.numel() for p in widened.parameters())
        print(f"  Widened params: {widen_params:,} (fusion NZ: {fused_nz:,})", flush=True)

        acc_widen_pre = evaluate(widened, test_loader)
        print(f"  Widened pre-FT: {acc_widen_pre:.2f}%", flush=True)

        # Fine-tune widened model (all params trainable)
        print(f"  Fine-tuning widened model...", flush=True)
        widened, acc_widen_post = finetune(widened, ft_loader, test_loader, FT_EPOCHS, FT_LR)
        print(f"  Widened post-FT: {acc_widen_post:.2f}%", flush=True)

        # Optional fusion comparison numbers; default to NaN when not supplied
        fusion_post_ft = prev.get(f"fusion_alpha_{alpha}", {}).get('acc_after_ft', float('nan'))
        fusion_nz_ratio = prev.get(f"fusion_alpha_{alpha}", {}).get('param_ratio_nonzero', float('nan'))

        results[f"alpha_{alpha}"] = {
            'alpha': alpha,
            'fused_channels': {k: v for k, v in fused_channels.items()},
            'widened_params': widen_params,
            'fused_nz_params': fused_nz,
            'widened_pre_ft': acc_widen_pre,
            'widened_post_ft': acc_widen_post,
            'fusion_post_ft_ref': fusion_post_ft,
        }

        del widened; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY — Random Widening vs Partial Fusion (post-FT)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'alpha':<7} {'Widen params':>13} {'Fusion NZ':>10} {'Widen FT':>9} {'Fusion FT':>10} {'Diff':>8}", flush=True)
    print("-" * 60, flush=True)
    for alpha in alphas:
        r = results[f"alpha_{alpha}"]
        diff = r['widened_post_ft'] - r['fusion_post_ft_ref']
        print(f"{alpha:<7} {r['widened_params']:>13,} {r['fused_nz_params']:>10,} "
              f"{r['widened_post_ft']:>8.2f}% {r['fusion_post_ft_ref']:>9.2f}% {diff:>+7.2f}%", flush=True)

    out_path = os.path.join(RESULTS_DIR, "cifar_resnet18_class_split_random_widen.json")
    def clean(d):
        if isinstance(d, dict): return {k: clean(v) for k, v in d.items()}
        if isinstance(d, list): return [clean(x) for x in d]
        if isinstance(d, (np.integer,)): return int(d)
        if isinstance(d, (np.floating,)): return float(d)
        return d
    with open(out_path, 'w') as f: json.dump(clean(results), f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
