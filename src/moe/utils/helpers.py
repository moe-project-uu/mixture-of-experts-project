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


def plot_expert_probs_by_class(class_expert_mean, class_names=None, ff_layer="SoftMoE"):
    """
    Grouped bars: x-axis are classes; for each class, E bars (one per expert)
    class_expert_mean: np.ndarray of shape (num_classes, num_experts)
    """
    import numpy as np
    import matplotlib.pyplot as plt

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
    import numpy as np
    import matplotlib.pyplot as plt

    C, E = class_expert_mean.shape
    if class_names is None or len(class_names) != C:
        class_names = [str(i) for i in range(C)]

    plt.figure(figsize=(E*0.6 + 3, C*0.4 + 2))
    plt.imshow(class_expert_mean, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="mean gating prob")
    plt.yticks(range(C), class_names)
    plt.xticks(range(E), [f"E{e}" for e in range(E)])
    plt.title(f"{ff_layer}: Expert probabilities heatmap (class × expert)")
    plt.tight_layout()
    plt.show()


#### ----- End of PlottingHelper functions ----- ####