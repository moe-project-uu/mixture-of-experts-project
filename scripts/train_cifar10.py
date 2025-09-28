"""
Train a CNN classifier for the CIFAR-10 dataset
This is a dense baseline comparison to the MoE model
Components of the script are:
1. Data loading
2. Model definition
3. Training loop
4. Evaluation
5. Saving the model
6. Loading the model
7. Testing the model
"""

#imports
import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets
import torchvision.transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np


#hyperparameters
BATCH_SIZE = 128
NUM_WORKERS = 2
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

###########----------------------------------###########
#dataset and data loader
###########----------------------------------###########
#transforms with augmentations, using CIFAR-10’s standard normalization
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
    # Optional augs:
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomCrop(32, padding=4),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)

#train and test data loaders are used to iterate over the dataset in batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


###########----------------------------------###########
#Model definition: small CNN for CIFAR-10
###########----------------------------------###########

#size of CIFAR-10 images is 32x32x3
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # common order: conv -> bn -> relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class dense_block(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class cnn_classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels=in_channels, out_channels=32,  kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_block(in_channels=32,          out_channels=64,  kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 32 -> 16

        self.conv3 = conv_block(in_channels=64,          out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv_block(in_channels=128,         out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 16 -> 8

        self.flatten = nn.Flatten()
        self.fc1 = dense_block(in_features=256*8*8, out_features=128)
        self.fc2 = dense_block(in_features=128,     out_features=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



##########----------------------------------##########
#Training + Evaluation Loops
##########----------------------------------##########
#set seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#model
model = cnn_classifier(in_channels=3, out_channels=10)
model.to(DEVICE)

#loss function
loss_fn = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#helper: calculate accuracy from logits
def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct, targets.size(0)


# per-epoch training + evaluation with metrics printing
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
    for data, targets in pbar:
        data = data.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        scores = model(data)
        loss = loss_fn(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate train metrics
        train_loss_sum += loss.item()
        correct, total = accuracy_from_logits(scores, targets)
        train_correct += correct
        train_total += total

    train_loss_avg = train_loss_sum / len(train_loader)
    train_acc = train_correct / train_total

    # ---- EVAL ----
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            scores = model(data)
            loss = loss_fn(scores, targets)

            val_loss_sum += loss.item()
            correct, total = accuracy_from_logits(scores, targets)
            val_correct += correct
            val_total += total

    val_loss_avg = val_loss_sum / len(test_loader)
    val_acc = val_correct / val_total

    # ---- PRINT per-epoch summary ----
    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"train_loss={train_loss_avg:.4f} train_acc={train_acc*100:.2f}% | "
        f"val_loss={val_loss_avg:.4f} val_acc={val_acc*100:.2f}%"
    )
    # save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(
            {'model': model.state_dict(), 'val_acc': best_val_acc},
            'checkpoints/cifar10_cnn.pt'
        )
        print(f"Saved checkpoint: val_acc={best_val_acc*100:.2f}%")