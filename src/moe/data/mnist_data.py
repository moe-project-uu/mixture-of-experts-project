# src/moe/data/cifar10_data.py
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as T

def build_mnist_train_val_test(
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
    Build MNIST dataloaders with train/val/test splits.
    Returns (train_loader, val_loader, test_loader, meta)
    """

    ## TODO: implement augmentation
    ## TODO: implement normalization

    # --- transforms ---
    train_tf = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.flatten()) # Flattens the C x H x W tensor to a 1D vector
    ])
    # for val + test (no agumentation)
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.flatten()) # Flattens the C x H x W tensor to a 1D vector
    ])  

    # --- raw datasets ---
    full_train = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.MNIST(
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
        "sizes": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
    }

    return train_loader, val_loader, test_loader, meta

def create_class_dataloaders(test_dataset, class_num):    
    # 1. Group test dataset indices by their label (0-9)
    seperated_data_indices = [[] for _ in range(class_num)]
    for idx, (data, label) in enumerate(test_dataset):
        seperated_data_indices[label].append(idx)
        
    # 2. Create Subsets and DataLoaders
    test_subsets = []
    test_loaders = []

    for label in range(10):
        indices = seperated_data_indices[label]
        
        # Create a Subset of the original dataset using the collected indices
        subset = Subset(test_dataset, indices)
        test_subsets.append(subset)
        
        # Create a DataLoader for the subset. Using len(indices) as batch_size 
        loader = DataLoader(
            dataset=subset, 
            batch_size=len(indices), 
            shuffle=False
        )
        test_loaders.append(loader)
    
    return test_loaders