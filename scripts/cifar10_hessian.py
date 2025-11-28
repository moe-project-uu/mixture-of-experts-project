# scripts/cifar10_hessian.py
#
# Hessian-based generalization & curvature analysis for CIFAR-10 models.
# Expects that train_cifar10.py has produced:
#   - checkpoints/.../model.pt
#   - checkpoints/.../summary.json
#
# Usage example:
#   python scripts/cifar10_hessian.py \
#       --ckpt_dir checkpoints/Dense/E50 \
#       --data_dir ./data \
#       --num_train 2000 \
#       --num_test 2000 \
#       --curv_eps 0.5 \
#       --curv_points 21

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from pyhessian import hessian as PyHessian

from moe.data.cifar10_data import build_cifar10_train_val_test
from moe.utils.helpers import make_subset_loader, build_model_from_summary


# ---------------------------------------------------------------------
# Curvature helper (loss vs α along a given direction)
# ---------------------------------------------------------------------
def compute_curvature_along_direction(model, criterion, dataloader, device,
                                      direction_vecs, alphas):
    """
    GOAL: Compute the loss L(θ + α v) for a range of α values.

    Args:
      model          : nn.Module, already loaded with trained weights.
      criterion      : loss function, e.g. nn.CrossEntropyLoss().
      dataloader     : DataLoader over a subset (train or test).
      device         : "cuda" or "cpu".
      direction_vecs : list of tensors, same shapes/order as model.parameters(),
                       representing a single direction v in parameter space
                       (e.g. top Hessian eigenvector).
      alphas         : 1D numpy array of scalars, e.g. np.linspace(-eps, eps, N).

    Returns:
      losses : list[float] of length len(alphas),
               each is the average loss over the dataloader at θ + α v.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    # Backup original parameters θ
    orig = [p.data.clone() for p in params]

    # Normalize direction v to unit norm over all parameters
    with torch.no_grad():
        total_norm_sq = sum((v.to(device) ** 2).sum() for v in direction_vecs)
        norm = total_norm_sq.sqrt().item()
        direction_vecs = [v.to(device) / norm for v in direction_vecs]

    losses = []

    for alpha in alphas:
        alpha = float(alpha)

        # θ' = θ + α v
        with torch.no_grad():
            for p, p0, v in zip(params, orig, direction_vecs):
                p.data = p0 + alpha * v

        # Compute average loss on the subset at θ'
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)
                total_n += x.size(0)

        losses.append(total_loss / total_n)

    # Restore original parameters θ
    with torch.no_grad():
        for p, p0 in zip(params, orig):
            p.data = p0

    return losses


# ---------------------------------------------------------------------
# Hessian + curvature wrapper
# ---------------------------------------------------------------------
def compute_hessian_and_curvature(model, criterion, dataloader, device,
                                  top_n=5,
                                  curvature_alphas=None,
                                  use_top_eigenvector_for_curvature=True):
    """
    Compute:
      - top eigenvalues (and eigenvectors)
      - trace
      - ESD (density via SLQ)
      - optionally, curvature (loss vs α) along a direction in parameter space
        (by default, the top eigenvector).

    Args:
      model     : nn.Module (weights already loaded).
      criterion : loss function (e.g. nn.CrossEntropyLoss()).
      dataloader: DataLoader over subset (train or test).
      device    : "cuda" or "cpu".
      top_n     : number of top eigenvalues to estimate.
      curvature_alphas :
          - None → skip curvature
          - 1D np.array → evaluate L(θ + α v) for α in curvature_alphas.
      use_top_eigenvector_for_curvature :
          - True  → v = top Hessian eigenvector
          - False → v = random direction in parameter space

    Returns:
      stats : dict with keys:
          "lambda_max"      : float
          "top_eigs"        : list[float] of length top_n
          "trace"           : float
          "density_eigs"    : np.ndarray
          "density_weights" : np.ndarray
        and, if curvature_alphas is not None:
          "curvature_alphas": np.ndarray
          "curvature_losses": list[float]
    """
    # Build PyHessian object for this dataloader
    h = PyHessian(model, criterion, dataloader=dataloader, cuda=(device == "cuda"))

    # 1) Top eigenvalues + eigenvectors (power iteration)
    top_eigs, top_vecs = h.eigenvalues(top_n=top_n)

    # 2) Trace via Hutchinson estimator
    trace = h.trace()

    # 3) Spectral density (ESD) via SLQ
    density_eigs, density_weights = h.density()

    # Base stats dict (always saved)
    stats = {
        "lambda_max": float(top_eigs[0]),              # largest eigenvalue
        "top_eigs": [float(x) for x in top_eigs],      # list of top eigenvalues
        "trace": float(trace),                         # Tr(H)
        "density_eigs": density_eigs,                  # np.ndarray
        "density_weights": density_weights,            # np.ndarray
    }

    # 4) Curvature along a direction (optional)
    if curvature_alphas is not None and len(curvature_alphas) > 0:
        if use_top_eigenvector_for_curvature:
            # top_vecs[0]: list of tensors matching model.parameters() shapes
            direction_vecs = top_vecs[0]
        else:
            # Alternative: random direction in parameter space
            params = [p for p in model.parameters() if p.requires_grad]
            direction_vecs = [torch.randn_like(p.data) for p in params]

        curv_losses = compute_curvature_along_direction(
            model=model,
            criterion=criterion,
            dataloader=dataloader,
            device=device,
            direction_vecs=direction_vecs,
            alphas=curvature_alphas,
        )

        stats["curvature_alphas"] = curvature_alphas
        stats["curvature_losses"] = curv_losses

    return stats


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory containing model.pt and summary.json"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to CIFAR-10 data root (same as training)"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=2000,
        help="Number of train samples used for Hessian estimation"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=2000,
        help="Number of test samples used for Hessian estimation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for Hessian dataloaders"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="num_workers for dataloaders"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top eigenvalues to estimate"
    )
    parser.add_argument(
        "--curv_eps",
        type=float,
        default=0.5,
        help="Curvature range: alphas will be in [-curv_eps, curv_eps]"
    )
    parser.add_argument(
        "--curv_points",
        type=int,
        default=21,
        help="Number of points in curvature grid (loss vs α)"
    )
    parser.add_argument(
        "--no_curvature",
        action="store_true",
        help="If set, skip curvature computation (only Hessian stats)."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load summary + checkpoint ---- #
    summary_path = os.path.join(args.ckpt_dir, "summary.json")
    model_path = os.path.join(args.ckpt_dir, "model.pt")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found at {summary_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pt not found at {model_path}")

    # Load JSON summary of model run
    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(f"Loaded summary from {summary_path}")
    print(f"FF_layer={summary['FF_layer']} | num_experts={summary.get('num_experts', 'N/A')}")

    # Rebuild model and load weights
    model = build_model_from_summary(summary, device)
    state = torch.load(model_path, map_location=device)
    # Saved in train_cifar10.py as {"model": model.state_dict(), "val_acc": ...}
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    # ---- build CIFAR-10 dataloaders ---- #
    # For Hessian & curvature, disable augmentation for determinism.
    train_loader, val_loader, test_loader, meta = build_cifar10_train_val_test(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        augment=False,
        drop_last=False,
        val_ratio=summary.get("val_ratio", 0.1),
        seed=summary.get("seed", 42),
    )

    # ---- make subsets for Hessian ---- #
    seed = summary.get("seed", 42)

    train_subset_loader = make_subset_loader(
        full_loader=train_loader,
        num_samples=args.num_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=seed,
    )
    test_subset_loader = make_subset_loader(
        full_loader=test_loader,
        num_samples=args.num_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=seed,
    )

    criterion = nn.CrossEntropyLoss()

    # curvature alphas for train set (used in slides)
    curvature_alphas = None
    if not args.no_curvature:
        curvature_alphas = np.linspace(
            -args.curv_eps, args.curv_eps, args.curv_points
        ).astype(np.float32)

    # ---- run PyHessian on train subset (with curvature) ---- #
    print("[HESSIAN] Computing train-set stats (and curvature)...")
    hessian_train = compute_hessian_and_curvature(
        model=model,
        criterion=criterion,
        dataloader=train_subset_loader,
        device=device,
        top_n=args.top_n,
        curvature_alphas=curvature_alphas,
        use_top_eigenvector_for_curvature=True,
    )
    print(f"  train λ_max = {hessian_train['lambda_max']:.4f}, "
          f"trace = {hessian_train['trace']:.4f}")

    # ---- run PyHessian on test subset (no curvature, usually enough) ---- #
    print("[HESSIAN] Computing test-set stats...")
    hessian_test = compute_hessian_and_curvature(
        model=model,
        criterion=criterion,
        dataloader=test_subset_loader,
        device=device,
        top_n=args.top_n,
        curvature_alphas=None,  # no curvature on test by default
        use_top_eigenvector_for_curvature=True,
    )
    print(f"  test  λ_max = {hessian_test['lambda_max']:.4f}, "
          f"trace = {hessian_test['trace']:.4f}")

    # ---- save results ---- #
    out_path = os.path.join(args.ckpt_dir, "hessian.pt")
    torch.save(
        {
            "summary": summary,
            "hessian": {
                "train": hessian_train,
                "test": hessian_test,
            },
        },
        out_path,
    )
    print(f"[HESSIAN] Saved Hessian stats (and curvature) to {out_path}")


if __name__ == "__main__":
    main()
