# src/moe/data/cifar10_data.py
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

CIFAR10_STATS = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std":  (0.2470, 0.2435, 0.2616),
    "num_classes": 10,
}

def build_cifar10_train_val_test(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    device: str = "cuda",
    augment: bool = True,
    drop_last: bool = False,
    val_ratio: float = 0.1,   # 10% of training data used for validation
    seed: int = 42,           # reproducible split
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Build CIFAR-10 dataloaders with train/val/test splits.
    Returns (train_loader, val_loader, test_loader, meta)
    """

    mean, std = CIFAR10_STATS["mean"], CIFAR10_STATS["std"]

    # --- transforms ---
    train_tf = (
        T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]) if augment else
        T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    )
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])  # for val + test

    # --- raw datasets ---
    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_tf
    )

    # --- split train into train/val ---
    train_size = int((1 - val_ratio) * len(full_train))
    val_size = len(full_train) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    # validation should use eval transforms (no augmentations)
    val_set.dataset.transform = eval_tf

    # --- dataloaders ---
    pin_memory = (device == "cuda")
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        drop_last=drop_last,
    )

    train_loader = DataLoader(train_set, shuffle=True,  **common)
    val_loader   = DataLoader(val_set,   shuffle=False, **common)
    test_loader  = DataLoader(test_set,  shuffle=False, **common)

    # --- metadata ---
    meta: Dict[str, Any] = {
        **CIFAR10_STATS,
        "sizes": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
    }

    return train_loader, val_loader, test_loader, meta
