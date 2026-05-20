"""
Sharpness-Aware Minimization (Foret, Kleiner, Mobahi, Neyshabur 2021).

Implementation follows the canonical reference (davda54/sam) used by the
SAM authors' published recipes. Wraps an inner optimizer (typically SGD)
and shares its param_groups, so any LR scheduler attached to the SAM
optimizer transparently controls the base optimizer's learning rate.

Standard recipe for ResNet18 / CIFAR10:
    rho = 0.05
    base = SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedule: CosineAnnealingLR(T_max=200), 200 epochs, bs=128

Training-loop usage (one SGD step = two forward-backward passes):

    optimizer.zero_grad()
    loss_fn(model(x), y).backward()
    optimizer.first_step(zero_grad=True)        # perturb w -> w + e
    disable_running_stats(model)                # avoid double-counting BN stats
    loss_fn(model(x), y).backward()             # gradient at w + e
    optimizer.second_step(zero_grad=True)       # restore w, step base optimizer
    enable_running_stats(model)
"""
import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        # Share param_groups with the base optimizer so any LR scheduler
        # attached to SAM controls the base optimizer's parameters too.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Adaptive SAM (ASAM, Kwon et al. 2021): scale perturbation
                # element-wise by |p|. Standard SAM keeps adaptive=False.
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, (
            "SAM.step() requires a closure that re-evaluates the loss; "
            "prefer calling first_step()/second_step() explicitly."
        )
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm


def disable_running_stats(model):
    """Stop BatchNorm running-stat updates so SAM's second forward pass does
    not double-count batch statistics. Stash original momentum on the module
    so it can be restored by :func:`enable_running_stats`.
    """
    def _disable(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.backup_momentum = m.momentum
            m.momentum = 0
    model.apply(_disable)


def enable_running_stats(model):
    def _enable(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum
    model.apply(_enable)
