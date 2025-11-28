# src/moe/utils/helpers.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Sequence, Dict, Any
from torch.utils.data import DataLoader, Subset
import torch
from moe.models.backbones import FeatureBackbone
from moe.heads.factory import build_head



#### ----- Helper functions ----- ####
# ---------------------------------------------------------------------
# Model reconstruction helper function (used in cifar10_hessian.py)
# ---------------------------------------------------------------------

def build_model_from_summary(summary, device):
    """
    Rebuild the CIFAR-10 model so that its strucutre is exactly as in train_cifar10.py using summary.json.
    We later load the state_dict parameters into this reconstructed model.
    """
    ff_layer = summary["FF_layer"]
    backbone = FeatureBackbone().to(device)

    if ff_layer == "Dense":
        head = build_head(
            "Dense",
            in_dim=backbone.output_dim,
            width=summary["ff_width"],
            num_classes=10,
        ).to(device)

    elif ff_layer == "SoftMoE":
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

    elif ff_layer == "SparseMoE":
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
        raise ValueError(f"Unknown FF_layer: {ff_layer}")

    class Classifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x))

    model = Classifier(backbone, head).to(device)
    return model


#Subset DataLoader helper function (used in cifar10_hessian.py)
def make_subset_loader(full_loader, num_samples, batch_size, num_workers, device, seed=42):
    """
    Description: Create a deterministic subset DataLoader with num_samples from full_loader.dataset.
    USAGE: We use this in the context of Hessian analysis to create a subset of the dataset for Hessian estimation.

    inputs: full_loader: DataLoader
    + other arguments as needed
    output = subset_loader: DataLoader
    """
    ds = full_loader.dataset
    n = len(ds)

    if (num_samples is None) or (num_samples >= n):
        #if num_samples is None or greater than the dataset size, use all samples
        indices = torch.arange(n)
    else:
        g = torch.Generator().manual_seed(seed)
        #otherwise, use a random subset of the dataset
        indices = torch.randperm(n, generator=g)[:num_samples]

    subset = Subset(ds, indices)

    subset_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    return subset_loader



#### ----- PlottingHelper functions ----- ####
def plot_expert_utilization(
    history: Dict[str, Any],
    ff_layer: str,
    epochs_to_bar: Sequence[int] = (0, 50, 99),
) -> None:
    """
    Plot expert utilization:
      1) Line plot of mean routing probability per expert across epochs.
      2) Per-epoch bar charts ("one bin per expert") at selected epoch indices.

    Expects `history["util_per_epoch"]` = list of arrays shaped (num_experts,)
    This generalizes to Soft/Hard/Sparse MoE if you log the same metric
    (e.g., mean one-hot selection for Hard, mean normalized top-k for Sparse).

    Args:
        history: dict produced by training (must contain "util_per_epoch").
        ff_layer: name of the head ("SoftMoE", "HardMoE", "SparseMoE", etc.). Used for titles only.
        epochs_to_bar: epoch indices (0-based) for the bar charts.
    """
    util_list = history.get("util_per_epoch", [])
    if not util_list:
        print("No utilization recorded (history['util_per_epoch'] missing or empty).")
        return

    util = np.stack(util_list, axis=0)  # (num_epochs, num_experts)
    num_epochs, num_experts = util.shape

    # (A) Line plot — evolution over epochs
    plt.figure(figsize=(7, 4))
    for i in range(num_experts):
        plt.plot(util[:, i], label=f"expert {i}")
    plt.xlabel("epoch")
    plt.ylabel("mean routing prob U_i")
    plt.title(f"{ff_layer}: Expert Utilization over epochs")
    plt.legend(ncol=2)
    plt.show()

    # (B) Bars at chosen epochs (clamped to available range)
    picked = [e for e in epochs_to_bar if 0 <= e < num_epochs]
    for e in picked:
        vals = util[e]  # (num_experts,)
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(num_experts), vals)
        plt.xticks(np.arange(num_experts), [f"E{i}" for i in range(num_experts)])
        plt.ylim(0, 1)
        plt.ylabel("mean routing prob U_i")
        plt.xlabel("expert")
        plt.title(f"{ff_layer}: Expert mean probabilities at epoch index {e}")
        plt.show()


def plot_gating_entropy(
    history: Dict[str, Any],
    ff_layer: str,
) -> None:
    """
    Plot gating entropy (routing sharpness) over training epochs.

    Expects `history["entropy_per_epoch"]` = list of floats, one per epoch.
    Generalizes to any MoE variant if entropy is logged the same way.

    Args:
        history: dict with key "entropy_per_epoch"
        ff_layer: model head name ("SoftMoE", "HardMoE", etc.) for titles
    """
    H = np.array(history.get("entropy_per_epoch", []))
    if H.size == 0:
        print("No entropy recorded (history['entropy_per_epoch'] missing or empty).")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(H, color="tab:blue")
    plt.xlabel("epoch")
    plt.ylabel("entropy  H = -Σ p log p")
    plt.title(f"{ff_layer}: Gating Entropy over epochs")
    plt.grid(alpha=0.3)
    plt.show()


def plot_expert_probs_by_class(class_expert_mean, class_names=None, ff_layer="SoftMoE"):
    """
    input: class_expert_mean: np.ndarray of shape (num_classes, num_experts)
    output: matplotlib figure
    Grouped bars: x-axis are classes; for each class, E bars (one per expert)
    class_expert_mean: np.ndarray of shape (num_classes, num_experts)
    """

    C, E = class_expert_mean.shape
    if class_names is None or len(class_names) != C:
        class_names = [str(i) for i in range(C)]

    x = np.arange(C)
    width = 0.8 / E  # total group width ~0.8

    plt.figure(figsize=(max(8, C*1.0), 5))
    for e in range(E):
        plt.bar(x + (e - (E-1)/2) * width, class_expert_mean[:, e], width=width, label=f"expert {e}")
    plt.xticks(x, class_names, rotation=30)
    plt.ylim(0, 1)
    plt.ylabel("mean gating prob")
    plt.xlabel("class")
    plt.title(f"{ff_layer}: Mean expert probabilities by class")
    plt.legend(ncol=min(E, 4))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_expert_probs_heatmap(class_expert_mean, class_names=None, ff_layer="SoftMoE"):
    """
    Heatmap view: rows = classes, cols = experts
    """

    C, E = class_expert_mean.shape
    if class_names is None or len(class_names) != C:
        class_names = [str(i) for i in range(C)] # default to class indices if not provided

    plt.figure(figsize=(E*0.6 + 3, C*0.4 + 2))
    plt.imshow(class_expert_mean, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="mean gating prob")
    plt.yticks(range(C), class_names)
    plt.xticks(range(E), [f"E{e}" for e in range(E)])
    plt.title(f"{ff_layer}: Expert probabilities heatmap (class × expert)")
    plt.tight_layout()
    plt.show()

def plot_expert_utilization_snapshot(util_vec, ff_layer="SoftMoE", label="test (best-val model)"):
    """
    Plots a single post-training utilization snapshot.
    util_vec: 1D array (num_experts,)
    """
    E = len(util_vec)
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(E), util_vec)
    plt.xticks(np.arange(E), [f"E{i}" for i in range(E)])
    plt.ylim(0, 1)
    plt.ylabel("mean routing prob U_i")
    plt.xlabel("expert")
    plt.title(f"{ff_layer}: Utilization snapshot — {label}")
    plt.tight_layout()
    plt.show()

def plot_expert_load_over_epochs(history, ff_layer: str):
    """
    Expects history['load_per_epoch']: list of arrays (E,)
    For sparse moe, this is fraction of samples selecting each expert.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    load_list = history.get("load_per_epoch", [])
    if not load_list:
        print("No load recorded (history['load_per_epoch'] missing or empty).")
        return

    L = np.stack(load_list, axis=0)  # (epochs, E)
    num_epochs, num_experts = L.shape

    plt.figure(figsize=(7, 4))
    for i in range(num_experts):
        plt.plot(L[:, i], label=f"E{i}")
    plt.xlabel("epoch")
    plt.ylabel("fraction of samples (load)")
    plt.title(f"{ff_layer}: Expert load over epochs")
    plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_expert_load_snapshot(load_vec, ff_layer="SparseMoE", label="test (best-val model)"):
    """
    Bar chart of a single load snapshot: load_vec shape (E,)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    E = len(load_vec)
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(E), load_vec)
    plt.xticks(np.arange(E), [f"E{i}" for i in range(E)])
    plt.ylim(0, 1)
    plt.ylabel("fraction of samples (load)")
    plt.xlabel("expert")
    plt.title(f"{ff_layer}: Load snapshot — {label}")
    plt.tight_layout()
    plt.show()


#### ----- End of PlottingHelper functions ----- ####