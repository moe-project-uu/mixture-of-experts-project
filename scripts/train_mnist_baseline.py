#modified from the deeplearning course at Uppsala University
"""
Train a CNN classifier for the MNIST dataset which is under data/MNIST
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
##
## Will need to rewrite this to include the new refactor and validation datasets
##
##
##
##
##

#imports
import numpy as np 
import random
import torch
import torch.nn as nn
from moe.data import mnist_data
from moe.models.MNIST_CNN import MNIST_CNN
from moe.heads.factory import build_head
from moe.training.training import training_loop
import time 
import os
import torch.optim as optim
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001


def train_mnist():
    ##########----------------------------------###########
    #Data loading
    ##########----------------------------------###########
    train, val, test, meta = mnist_data.build_mnist_train_val_test()

    ##########----------------------------------###########
    #Build the model
    ##########----------------------------------###########
    backbone = MNIST_CNN().to(DEVICE)

    head = build_head(
                "Dense",                    # "Dense"
                in_dim=512,       #512 
                width=32,              
                num_classes=10
                ).to(DEVICE)

    class Classifier(nn.Module):
        def __init__(self, backbone, head): 
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x, return_gate=False):
            h = self.backbone(x)                      # (B, 512)
            return self.head(h, return_gate=return_gate)

    model = Classifier(backbone, head).to(DEVICE)

    ##########----------------------------------###########
    #Training the model
    ##########----------------------------------###########
    # --- loss & optimizer & scheduler ---
    criterion = nn.CrossEntropyLoss()  # you can try label_smoothing=0.1
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)  # good for 50 epochs

    history, (best_train_acc, best_train_epoch), (best_val_acc, best_val_epoch) = training_loop(
        train_loader = train, 
        val_loader = val,
        num_epochs = EPOCHS, 
        model = model, 
        optimizer = optimizer, 
        criterion = criterion,
        scheduler = scheduler,
        FF_layer_type = "Dense",
        DEVICE = DEVICE,
        ckpt_model_path = "",
        softmoe_load_balance_required = False,
        experts=10
    )

        # os.makedirs('checkpoints', exist_ok=True)
        # torch.save(
        #     {'model': model.state_dict(), 'test_accuracy': test_accuracy, 'test_loss': test_loss, 
        #     'training_accuracy': training_accuracy, 'training_loss': training_loss, 'total_time': total_time,
        #     'optimizer': optimizer.state_dict()},
        #     'checkpoints/MNIST.pt'
        # )
        # print(f"Saved checkpoint to checkpoints/MNIST.pt")

if __name__=="__main__":
    train_mnist()