#!/usr/bin/env python3
"""
ResNet18 even/odd class-split partial fusion with cosine BN stat initialization.
================================================================================
Trains two ResNet18 specialists on CIFAR-10 (model A on even classes, model B on
odd classes), then partially fuses them across alpha values using cosine
initialization for batch-norm running statistics. Each fusion is masked-fine-tuned
and compared against a single-model fine-tuned baseline.
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
    split_a, split_b, split_ft = [], [], []
    for c in range(10):
        idx = np.where(targets == c)[0].copy(); np.random.shuffle(idx)
        n = len(idx); n_ft = int(0.05*n); n_major = int(0.92*n); n_minor = int(0.03*n)
        ft_idx = idx[:n_ft]; rest = idx[n_ft:]
        if c in EVEN_CLASSES:
            split_a.extend(rest[:n_major].tolist()); split_b.extend(rest[n_major:n_major+n_minor].tolist())
        else:
            split_b.extend(rest[:n_major].tolist()); split_a.extend(rest[n_major:n_major+n_minor].tolist())
        split_ft.extend(ft_idx.tolist())
    return train_aug, test_set, split_ft

def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total

def create_nonzero_masks(model):
    return {name: (param.data.abs() > 0).float() for name, param in model.named_parameters()}

def count_nonzero_params(model):
    return int(sum((p.data.abs() > 0).sum().item() for p in model.parameters()))

def finetune_masked(model, ft_loader, test_loader, masks, epochs=30, lr=0.001):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = evaluate(model, test_loader); best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for x, y in ft_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward()
            for name, param in model.named_parameters():
                if name in masks and param.grad is not None: param.grad.data *= masks[name]
            optimizer.step()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks: param.data *= masks[name]
        scheduler.step()
        acc = evaluate(model, test_loader)
        if acc > best_acc: best_acc = acc; best_state = copy.deepcopy(model.state_dict())
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            print(f"    FT {epoch+1}/{epochs}: {acc:.2f}% (best={best_acc:.2f}%)", flush=True)
    model.load_state_dict(best_state)
    return model, best_acc

def main():
    results = {}
    print("=" * 70, flush=True)
    print("Exp 25c: Even/Odd + Cosine BN Stats + Masked FT", flush=True)
    print("=" * 70, flush=True)

    train_aug, test_set, split_ft = get_data_and_splits()
    ft_loader = DataLoader(Subset(train_aug, split_ft), batch_size=FT_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model_a = ResNet18(num_classes=10).to(device)
    model_a.load_model(os.path.join(CKPT_DIR, "resnet18_evenodd_a.checkpoint"))
    model_b = ResNet18(num_classes=10).to(device)
    model_b.load_model(os.path.join(CKPT_DIR, "resnet18_evenodd_b.checkpoint"))
    acc_a = evaluate(model_a, test_loader); acc_b = evaluate(model_b, test_loader)
    n_params_single = sum(p.numel() for p in model_a.parameters())
    print(f"Model A: {acc_a:.2f}%, Model B: {acc_b:.2f}%", flush=True)

    # Baseline reference: best single-model fine-tuned accuracy
    best_single_ft = 86.73
    print(f"Best single FT ref: {best_single_ft:.2f}%", flush=True)
    results['best_single_ft_ref'] = best_single_ft

    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    for alpha in alphas:
        print(f"\n  alpha={alpha} (cosine BN)", flush=True)
        t0 = time.time()
        fm = PartialFusion(alphas=alpha, combine_costs=True, pgd=True, tied_permutations=True, warmstart=True)
        fused = FusionModel(copy.deepcopy(model_a), copy.deepcopy(model_b), fm,
                            lambdas=[0.5, 0.5], running_stats_init='cosine')
        fusion_time = time.time() - t0
        nz = count_nonzero_params(fused)
        nz_ratio = nz / n_params_single

        acc_pre = evaluate(fused, test_loader)
        print(f"    Pre-FT: {acc_pre:.2f}% (NZ={nz:,}, {nz_ratio:.2f}x, fusion={fusion_time:.0f}s)", flush=True)

        masks = create_nonzero_masks(fused)
        fused, acc_post = finetune_masked(fused, ft_loader, test_loader, masks, FT_EPOCHS, FT_LR)
        print(f"    Post-FT: {acc_post:.2f}% (delta={acc_post-acc_pre:+.2f}%)", flush=True)

        results[f"fusion_alpha_{alpha}"] = {
            'alpha': alpha, 'fusion_time': fusion_time,
            'nonzero_params': nz, 'param_ratio_nonzero': round(nz_ratio, 3),
            'acc_before_ft': acc_pre, 'acc_after_ft': acc_post,
        }
        del fused; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY — Cosine BN + Masked FT", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'alpha':<7} {'pre-FT':>8} {'post-FT':>9} {'NZ ratio':>9} {'vs single FT':>13}", flush=True)
    print("-" * 48, flush=True)
    for alpha in alphas:
        r = results[f"fusion_alpha_{alpha}"]
        d = r['acc_after_ft'] - best_single_ft
        print(f"{alpha:<7} {r['acc_before_ft']:>7.2f}% {r['acc_after_ft']:>8.2f}% {r['param_ratio_nonzero']:>8.2f}x {d:>+12.2f}%", flush=True)

    out_path = os.path.join(RESULTS_DIR, "cifar_resnet18_class_split_cosine_bn.json")
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
