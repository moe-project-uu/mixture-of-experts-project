# scripts/cifar10_efficiency.py
#
# Inference-time efficiency benchmark for CIFAR-10 checkpoints:
#   - Parameter count / parameter size (MB)
#   - Peak GPU memory during forward pass (MB)
#   - Latency (ms/batch) and throughput (images/sec)
#
# Usage:
#   python scripts/cifar10_efficiency.py --ckpt_dir checkpoints/cifar/Dense/E50
#   python scripts/cifar10_efficiency.py --ckpt_dir checkpoints/cifar/SoftMoE/E50-X8
#   python scripts/cifar10_efficiency.py --ckpt_dir checkpoints/cifar/SparseMoE/E50-X8-K2

import os
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from moe.data.cifar10_data import build_cifar10_train_val_test
from moe.models.backbones import FeatureBackbone
from moe.heads.factory import build_head


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def params_mb_fp32(model: nn.Module) -> float:
    # fp32 => 4 bytes per parameter
    return (count_params(model) * 4) / (1024 ** 2)


def build_model_from_summary(summary: dict, device: str) -> nn.Module:
    """
    Reconstruct the exact model architecture used in training from summary.json.
    Mirrors scripts/train_cifar10.py logic.
    """
    FF_LAYER = summary["FF_layer"]
    backbone = FeatureBackbone().to(device)

    if FF_LAYER == "Dense":
        head = build_head(
            "Dense",
            in_dim=backbone.output_dim,
            width=summary["ff_width"],
            num_classes=10,
        ).to(device)

    elif FF_LAYER == "SoftMoE":
        head = build_head(
            "SoftMoE",
            in_dim=backbone.output_dim,
            num_classes=10,
            num_experts=summary["num_experts"],
            hidden_mult=summary["hidden_mult"],
            temperature=summary["temperature"],
            dropout_p=summary["dropout_p"],
            gate_input_dropout=summary["gate_input_dropout"],
            gate_logits_dropout=summary["gate_logits_dropout"],
        ).to(device)

    elif FF_LAYER == "SparseMoE":
        head = build_head(
            "SparseMoE",
            in_dim=backbone.output_dim,
            num_classes=10,
            num_experts=summary["num_experts"],
            hidden_mult=summary["hidden_mult"],
            temperature=summary["temperature"],
            dropout_p=summary["dropout_p"],
            gate_input_dropout=summary["gate_input_dropout"],
            gate_logits_dropout=summary["gate_logits_dropout"],
            importance_coef=summary["sparsemoe_importance_coef"],
            load_coef=summary["sparsemoe_load_coef"],
            k=summary["sparsemoe_k"],
        ).to(device)
    else:
        raise NotImplementedError(f"Unknown FF_layer: {FF_LAYER}")

    class Classifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            h = self.backbone(x)
            out = self.head(h, return_gate=False)
            # Safety: if a head returns a tuple/list even with return_gate=False
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

    return Classifier(backbone, head).to(device)


@torch.no_grad()
def measure_peak_gpu_memory_mb(model: nn.Module, loader, device: str, num_batches: int = 20) -> float | None:
    """
    Returns peak allocated GPU memory (MB) for a few inference batches.
    If running on CPU, returns None.
    """
    if device != "cuda":
        return None

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    it = iter(loader)
    for _ in range(num_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        _ = model(x)

    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024 ** 2)


@torch.no_grad()
def benchmark_latency_throughput(
    model: nn.Module,
    loader,
    device: str,
    warmup_batches: int = 30,
    timed_batches: int = 200,
) -> dict:
    """
    Measures batch latency and throughput.
    Returns mean/std/median latency per batch (ms) and images/sec.
    """
    model.eval()

    # Warmup
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        _ = model(x)

    # Timed
    times = []
    it = iter(loader)
    bs = loader.batch_size if hasattr(loader, "batch_size") and loader.batch_size is not None else None

    for _ in range(timed_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break

        x = x.to(device, non_blocking=True)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times, dtype=np.float64)
    if len(times) == 0:
        raise RuntimeError("No timed batches were recorded. Check your dataloader.")

    mean_s = float(times.mean())
    std_s = float(times.std())
    median_s = float(np.median(times))

    # throughput: images/sec based on mean batch time
    if bs is None:
        # fallback: use actual tensor batch size from last x
        bs = int(x.shape[0])
    imgs_per_sec = float(bs / mean_s)

    return {
        "batch_size": int(bs),
        "latency_mean_ms_per_batch": mean_s * 1e3,
        "latency_std_ms_per_batch": std_s * 1e3,
        "latency_median_ms_per_batch": median_s * 1e3,
        "throughput_images_per_sec": imgs_per_sec,
        "num_timed_batches": int(len(times)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="Path containing summary.json and model.pt")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="Set 0 for benchmarking (avoids timing the data pipeline).")
    ap.add_argument("--warmup_batches", type=int, default=30)
    ap.add_argument("--timed_batches", type=int, default=200)
    ap.add_argument("--peakmem_batches", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary_path = os.path.join(args.ckpt_dir, "summary.json")
    model_path = os.path.join(args.ckpt_dir, "model.pt")
    out_path = os.path.join(args.ckpt_dir, "efficiency.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary.json in {args.ckpt_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model.pt in {args.ckpt_dir}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Build model + load trained weights
    model = build_model_from_summary(summary, device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    # Data loaders (test only is enough)
    _, _, test_loader, meta = build_cifar10_train_val_test(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        augment=False,
        drop_last=False,
        val_ratio=summary.get("val_ratio", 0.1),
        seed=summary.get("seed", 42),
    )

    # Basic parameter stats
    n_params = count_params(model)
    param_mb = params_mb_fp32(model)

    # Peak GPU memory (inference)
    peak_mb = measure_peak_gpu_memory_mb(
        model, test_loader, device=device, num_batches=args.peakmem_batches
    )

    # Latency / throughput
    timing = benchmark_latency_throughput(
        model,
        test_loader,
        device=device,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
    )

    result = {
        "ckpt_dir": args.ckpt_dir,
        "device": device,
        "ff_layer": summary.get("FF_layer", None),
        "num_params": int(n_params),
        "param_size_mb_fp32": float(param_mb),
        "peak_gpu_mem_mb_allocated": (None if peak_mb is None else float(peak_mb)),
        "timing": timing,
    }

    print(json.dumps(result, indent=2))

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
