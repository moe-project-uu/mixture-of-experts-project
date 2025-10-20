from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def load_data(data_dir: str):
    # Define the transformation to convert to Tensor and flatten it
    flatten_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()) # Flattens the C x H x W tensor to a 1D vector
    ])


    # Download and load the training dataset with the new transformation
    train_dataset = datasets.MNIST(
        root = data_dir,
        train = True,
        transform = flatten_transform,
        download = True
    )

    # Download and load the test dataset with the new transformation
    test_dataset = datasets.MNIST(
        root = data_dir,
        train = False,
        transform = flatten_transform,
        download = True
    )
    
    return train_dataset, test_dataset

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