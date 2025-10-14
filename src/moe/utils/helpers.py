# src/moe/utils/helpers.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Sequence, Dict, Any

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

#### ----- End of PlottingHelper functions ----- ####