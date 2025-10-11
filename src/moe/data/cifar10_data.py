# src/data/cifar10_data.py
from typing import Dict, Any
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

CIFAR10_STATS = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std":  (0.2470, 0.2435, 0.2616),
    "num_classes": 10,
}

def build_cifar10_train_test(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    device: str = "cuda",
    augment: bool = True,
    drop_last: bool = False,
):
    mean, std = CIFAR10_STATS["mean"], CIFAR10_STATS["std"]

    train_tf = (
        T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]) if augment else
        T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    )
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    pin_memory = (device == "cuda")
    #Note: common is a dictionary of common arguments for the DataLoader
    common = dict( 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        drop_last=drop_last,
    )

    train_loader = DataLoader(train_set, shuffle=True, **common) 
    test_loader  = DataLoader(test_set,  shuffle=False, **common)

    meta: Dict[str, Any] = {
        **CIFAR10_STATS, 
        "sizes": {"train": len(train_set), "test": len(test_set)},
    }
    return train_loader, test_loader, meta
