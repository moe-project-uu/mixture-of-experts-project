import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict, Any

# --- 1. Data Preprocessing and Loading ---

class CUB200Dataset(Dataset):
    """
    Dataset Class for CUB-200-2011.
    It uses the bounding boxes to crop the birds since I am only doing
    classification right now, and birds are mostly not centered in the images.
    """

    def __init__(self, root_dir: str, train: bool = True, transform: Any = None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root_dir, 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root_dir, 'image_class_labels.txt'), sep=' ', names=['img_id', 'class_id'])
        train_test_split = pd.read_csv(os.path.join(self.root_dir, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
        bounding_boxes = pd.read_csv(os.path.join(self.root_dir, 'bounding_boxes.txt'), sep=' ', names=['img_id', 'x', 'y', 'width', 'height'])

        # Merge all metadata into a single DataFrame
        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        data = data.merge(bounding_boxes, on='img_id')

        # Filter for train or test set
        if self.train:
            self.data = data[data['is_training_img'] == 1]
        else:
            self.data = data[data['is_training_img'] == 0]

        # PyTorch datasets expect class labels to be 0-indexed
        self.data['class_id'] = self.data['class_id'] - 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.data.iloc[idx]
        
        filepath = os.path.join(self.root_dir, 'images', sample['filepath'])
        image = Image.open(filepath).convert('RGB')

        # Crop image using the bounding box
        x, y, width, height = sample['x'], sample['y'], sample['width'], sample['height']
        image = image.crop((x, y, x + width, y + height))

        if self.transform:
            image = self.transform(image)
        
        class_id = sample['class_id']
        return image, class_id

def get_dataloaders(root_dir: str, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns the training and validation/test DataLoaders.
    Applies standard ImageNet normalization and data augmentation.
    """
    # Pre-trained models expect 224x224 inputs
    image_size = (224, 224)
    
    # ImageNet stats
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.1)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CUB200Dataset(root_dir=root_dir, train=True, transform=train_transform)
    test_dataset = CUB200Dataset(root_dir=root_dir, train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader