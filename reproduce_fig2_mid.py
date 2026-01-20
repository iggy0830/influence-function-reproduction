# Meaning of the plot:
#   - Each dot = one training example (we only plot the top-k most influential ones)
#   - x-axis = ground-truth effect (obtained by LOO retraining)
#   - y-axis = influence function linear approximation
#   - Ideal case: points lie on the diagonal y = x
#
# Key formula:
#   I_up,loss(z, z_test) = - ∇L(z_test)^T H^{-1} ∇L(z)
#
# Removing one sample ≈ upweighting it by ε = -1/n
#   ΔL_pred ≈ -(1/n) * I_up,loss
#
# Key idea:
#   1) Compute IHVP only once:
#        s_test = H^{-1} ∇L(z_test)
#   2) For each training point, compute a dot product:
#        I_up,loss(z_i) = - s_test^T ∇L(z_i)
# ------------------------------------------------------------

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


# 0) Basic utility functions

def set_seed(seed: int = 0) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of parameter tensors into a single long vector."""
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_like(vec: torch.Tensor, params: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    """
    Unflatten a long vector back into a list of tensors with the same shapes
    as model parameters (used in HVP).
    """
    out: List[torch.Tensor] = []
    offset = 0
    for p in params:
        n = p.numel()
        out.append(vec[offset: offset + n].view_as(p))
        offset += n
    return out


@torch.no_grad()
def copy_model(model: nn.Module) -> nn.Module:
    """Deep-copy the model (used for LOO retraining)."""
    import copy
    return copy.deepcopy(model)


# 1) Model: Multiclass Logistic Regression on MNIST

class LogisticRegressionMNIST(nn.Module):
    """
    A simple linear model (multinomial logistic regression).
    """
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.linear(x)



# 2) Data utilities: Dataset -> full tensors (for deterministic full-batch LBFGS)

@torch.no_grad()
def dataset_to_tensor(
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a Dataset into full X, Y tensors.
    Used for full-batch LBFGS training / retraining.
    """
    xs, ys = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs, dim=0).to(device)
    Y = torch.cat(ys, dim=0).to(device)
    return X, Y


def loss_on_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss on a batch (mean reduction)."""
    logits = model(x)
    return F.cross_entropy(logits, y)


@torch.no_grad()
def loss_on_single(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Loss on a single example."""
    model.eval()
    return float(loss_on_batch(model, x, y).item())



# 3) Gradients & Hessian–Vector Product (HVP)

def grad_theta_single(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute and flatten the gradient ∇_θ L(x, y; θ̂).

    Notation mapping to the paper:
      v    = ∇_θ L(z_test, θ̂)
      g_i  = ∇_θ L(z_i, θ̂)
    """
    model.zero_grad(set_to_none=True)
    loss = loss_on_batch(model, x, y)
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)
    return flatten_tensors(grads).detach()


def hvp_single_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Hessian–vector product: (∇²_θ L_batch) v.

    Autograd trick:
      g  = ∇_θ L
      Hv = ∇_θ (g^T v)
    """
    params = list(model.parameters())

    model.zero_grad(set_to_none=True)
    loss = loss_on_batch(model, x, y)
    g = torch.autograd.grad(loss, params, create_graph=True)

    v_list = unflatten_like(v, params)
    gv = torch.zeros((), device=v.device)
    for gi, vi in zip(g, v_list):
        gv = gv + (gi * vi).sum()

    hv = torch.autograd.grad(gv, params, create_graph=False)
    return flatten_tensors(hv).detach()


# 4) LiSSA: approximate inverse Hessian–vector product (IHVP)

@dataclass
class LiSSAConfig:
    # t: recursion steps; r: number of independent runs (variance reduction)
    t: int
    r: int
    # scale: scaling factor for Neumann/Taylor expansion to ensure convergence
    scale: float
    # damping: damping term (typically matched to L2 regularization)
    damping: float
    # minibatch size used for HVP estimation
    hvp_batch_size: int


def lissa_inverse_hvp(
    model: nn.Module,
    train_dataset: Dataset,
    v: torch.Tensor,
    cfg: LiSSAConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Approximate:
      s ≈ H^{-1} v

    - LiSSA uses a Neumann/Taylor-style recursion
    - Each step uses a minibatch Hessian estimate

    Log explanation:
      "LiSSA run a/b, step c/d":
        - run: which independent repetition (we average over r runs)
        - step: recursion step within that run (total t steps)
    """
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.hvp_batch_size,
        shuffle=True,
        drop_last=True
    )

    def infinite_batches():
        while True:
            for bx, by in loader:
                yield bx.to(device), by.to(device)

    stream = infinite_batches()
    estimates: List[torch.Tensor] = []
    v = v.detach()

    for run in range(cfg.r):
        u = v.clone()  # initialize recursion
        for step in range(cfg.t):
            # (1) sample a minibatch
            x_b, y_b = next(stream)

            # (2) compute stochastic HVP
            hv = hvp_single_batch(model, x_b, y_b, u)

            # (3) add damping for numerical stability
            hv = hv + cfg.damping * u

            # (4) LiSSA recursion update
            u = v + (u - hv / cfg.scale)

            if (step + 1) % 500 == 0:
                print(f"LiSSA run {run+1}/{cfg.r}, step {step+1}/{cfg.t}")

        estimates.append(u / cfg.scale)

    # average over runs to reduce variance
    return torch.stack(estimates, dim=0).mean(dim=0)


# 5) ERM training & LOO retraining (LBFGS)

@dataclass
class LBFGSConfig:
    max_iter: int
    lr: float
    weight_decay: float


def train_fullbatch_lbfgs(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    cfg: LBFGSConfig,
) -> None:
    """
    Optimize ERM with full-batch LBFGS.

    - Used to train the base model θ̂
    - Used for LOO retraining to obtain θ̂_{-i}
    """
    model.train()
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.lr,
        max_iter=cfg.max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)

        # ERM loss
        loss = loss_on_batch(model, X, Y)

        # L2 regularization (the paper folds regularization into L)
        if cfg.weight_decay > 0:
            wd = torch.zeros((), device=X.device)
            for p in model.parameters():
                wd = wd + (p ** 2).sum()
            loss = loss + 0.5 * cfg.weight_decay * wd

        loss.backward()
        return loss

    opt.step(closure)


@torch.no_grad()
def find_misclassified_test_point(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a misclassified test example."""
    model.eval()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=-1)
        mask = pred.ne(y)
        if mask.any():
            idx = int(mask.nonzero(as_tuple=False)[0].item())
            return x[idx:idx+1], y[idx:idx+1]
    raise RuntimeError("No misclassified test point found.")


def loo_delta_loss_lbfgs(
    base_model: nn.Module,
    train_subset: Subset,
    removed_pos: int,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    base_loss: float,
    cfg: LBFGSConfig,
    device: torch.device,
) -> float:
    """
    Ground-truth effect (reference):
      ΔL_true = L(z_test; θ̂_{-i}) - L(z_test; θ̂)

    Procedure:
      - remove one training example
      - warm-start from θ̂
      - retrain with LBFGS
      - compare test loss
    """
    keep = [j for j in range(len(train_subset)) if j != removed_pos]
    loo_ds = Subset(train_subset, keep)

    model = copy_model(base_model).to(device)
    X_loo, Y_loo = dataset_to_tensor(loo_ds, device=device)
    train_fullbatch_lbfgs(model, X_loo, Y_loo, cfg)

    new_loss = loss_on_single(model, test_x, test_y)
    return new_loss - base_loss


# 6) Main experiment pipeline

@dataclass
class ExpConfig:
    seed: int
    train_subset_size: int
    top_k: int
    save_path: str = "fig2_mid_repro.png"


def main() -> None:
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ===== Step 0: hyperparameters =====
    WD = 1e-3  # L2 regularization strength (match IHVP damping)

    exp = ExpConfig(
        seed=0,
        train_subset_size=20000,  # subset size (controls compute cost)
        top_k=200,                # only run LOO on the most influential points
    )

    lissa_cfg = LiSSAConfig(
        t=3000,
        r=10,
        scale=10.0,
        damping=WD,
        hvp_batch_size=512,
    )

    base_lbfgs = LBFGSConfig(max_iter=300, lr=1.0, weight_decay=WD)
    loo_lbfgs  = LBFGSConfig(max_iter=200, lr=1.0, weight_decay=WD)

    # ===== Step 1: load MNIST and sample a training subset =====
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(exp.seed)
    subset_idx = torch.randperm(len(train_ds_full), generator=g)[:exp.train_subset_size]
    train_subset = Subset(train_ds_full, subset_idx.tolist())

    test_loader = DataLoader(test_ds, batch_size=512)

    # ===== Step 2: train the base ERM model to obtain θ̂ =====
    model = LogisticRegressionMNIST().to(device)
    X_train, Y_train = dataset_to_tensor(train_subset, device=device)
    train_fullbatch_lbfgs(model, X_train, Y_train, base_lbfgs)

    # ===== Step 3: pick a misclassified test point z_test =====
    test_x, test_y = find_misclassified_test_point(model, test_loader, device)
    print("Picked misclassified test point label:", int(test_y.item()))
    base_test_loss = loss_on_single(model, test_x, test_y)

    # ===== Step 4: compute v = ∇_θ L(z_test, θ̂) =====
    v = grad_theta_single(model, test_x, test_y).to(device)

    # ===== Step 5: compute s_test ≈ H^{-1} v using LiSSA =====
    s = lissa_inverse_hvp(model, train_subset, v, lissa_cfg, device)

    # ===== Step 6: compute influence for each training point =====
    influences = []
    for i in range(len(train_subset)):
        x_i, y_i = train_subset[i]
        x_i = x_i.unsqueeze(0).to(device)
        y_i = torch.tensor([y_i], device=device)

        g_i = grad_theta_single(model, x_i, y_i)
        I_i = -torch.dot(s, g_i).item()
        influences.append(I_i)

    # ===== Step 7: convert influence into predicted LOO loss change =====
    n = len(train_subset)
    pred_delta = [-(1.0 / n) * I for I in influences]

    # ===== Step 8: select top-k points =====
    topk = sorted(range(n), key=lambda i: abs(influences[i]), reverse=True)[:exp.top_k]
    print("Top-k selected:", len(topk))

    # ===== Step 9: run true LOO retraining for the selected top-k points =====
    actual_delta, pred_delta_topk = [], []
    for j, pos in enumerate(topk):
        d_true = loo_delta_loss_lbfgs(
            model, train_subset, pos,
            test_x, test_y, base_test_loss,
            loo_lbfgs, device
        )
        actual_delta.append(d_true)
        pred_delta_topk.append(pred_delta[pos])

        if (j + 1) % 10 == 0:
            print(f"LOO done {j+1}/{len(topk)}")

    # ===== Step 10: print diagnostics (correlation and fitted slope) =====

    x = np.array(actual_delta)
    y = np.array(pred_delta_topk)

    pearson = float(np.corrcoef(x, y)[0, 1])
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    print(f"Pearson corr: {pearson:.4f}")
    print(f"Fit line: y = {a:.4f} * x + {b:.4f}")

    # ===== Step 11: plot scatter  =====
    plt.figure()
    plt.scatter(actual_delta, pred_delta_topk, s=20)

    m = min(min(actual_delta), min(pred_delta_topk))
    M = max(max(actual_delta), max(pred_delta_topk))
    pad = 0.05 * (M - m + 1e-12)
    m -= pad
    M += pad

    plt.plot([m, M], [m, M])
    plt.xlabel("Actual diff in loss (LOO retraining)")
    plt.ylabel("Linear approx (-1/n * I_up,loss)")
    plt.title("Figure 2 (Mid) reproduction")
    plt.tight_layout()
    plt.savefig(exp.save_path, dpi=200)
    print(f"Saved: {exp.save_path}")


if __name__ == "__main__":
    main()


# Conclusion: The Pearson correlation is 0.9694, showing that influence function predictions closely match the actual LOO retraining results.

# *
# The plot is more scattered than the paper because  LiSSA is a stochastic approximation of the inverse Hessian vector product
# The noise could be reduced with larger batches or conjugate gradients，but the computation cost would be much higher

