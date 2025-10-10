import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import random
import time
import sys
import os
import matplotlib.pyplot as plt

#for the disk_memoize function
import pickle
import hashlib
from functools import wraps
from tqdm import tqdm

# Add scripts folder path so I can get load_mnist
repo_root = os.path.abspath(os.path.join("..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from scripts.MNIST.load_mnist import load_mnist

#just some basic stuff to set for reproducability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

#taken from APML project HT 2025 so we only need to load in the dataset once
#will make it faster for testing I hope
def disk_memoize(cache_dir="cache_mnist"):
    """
    Decorator for caching function outputs on disk.

    This utility is already implemented and should not be modified by students.
    It allows expensive computations to be stored and re-used across runs,
    based on the function arguments. If you call the same function again with
    the same inputs, it returns the cached results instead of recomputing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optionally force a fresh computation (ignores cache if True)
            force = kwargs.pop("force_recompute", False)

            # Make sure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Build a unique hash key from the function name and arguments
            func_name = func.__name__
            key = (func_name, args, kwargs)
            hash_str = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func_name}_{hash_str}.pkl")

            # Load the cached result if it exists (and recomputation is not forced)
            if not force and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            # Otherwise: compute the result, then cache it to disk
            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            return result
        
        return wrapper
    return decorator

#want to get data in a linear form for our first moe tests
@disk_memoize
def get_data(linear = True):
    #get the train and test data from the dataset
    xtrain,ytrain,xtest,ytest = load_mnist.load_mnist()
    #if we want to work with flattened/ linear input
    if linear:
        xtrain = torch.Tensor(xtrain).to(DEVICE)
        ytrain = torch.Tensor(ytrain).to(DEVICE)
        xtest = torch.Tensor(xtest).to(DEVICE)
        ytest = torch.Tensor(ytest).to(DEVICE)
    else:
        #converting to Tensors for easy PyTorch implementation and reshape for a CNN
        xtrain = torch.Tensor(xtrain).reshape(60000, 1,28,28).to(DEVICE)
        ytrain = torch.Tensor(ytrain).to(DEVICE)
        xtest = torch.Tensor(xtest).reshape(10000, 1,28,28).to(DEVICE)
        ytest = torch.Tensor(ytest).to(DEVICE)
    #first we want to put our data in a pytorch dataset so we can mini batch and enumerate through it later more easily
    train_dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
    test_dataset = torch.utils.data.TensorDataset(xtest, ytest)
    #Making a dataloader for this specific CNN which is a wrapper around the Dataset for easy use
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #make the batch size for the test DataLoader the size of the dataset for evaluation.
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = ytest.shape[0], shuffle=True)
    return train_loader, test_loader



class SoftmaxGating(nn.Module):
    def __init__(self, input_dim: int, expert_num: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=expert_num)
    
    def forward(self, x):
        return self.linear(x)

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        pass
    
    def forward(self, x):
        # TODO
        pass
    
class MoE_Layer(nn.Module):
    def __init__(self, input_dim: int, expert_num: int = 4, top_k: int = 2):
        self.expert_num = expert_num
        self.top_k = top_k
        
        # set-up router
        self.router = SoftmaxGating(input_dim, self.expert_num)
        
        # set-up 
        self.experts = nn.ModuleList([Expert(...) for i in range(self.expert_num)])
    
    def forward(self, x):
        gating_logits = F.softmax(self.router(x), dim=-1) # last dim is the experts
        top_k_logits, top_k_indices = torch.topk(gating_logits, self.k_top, dim=-1)
        
        for i, expert in enumerate(self.experts):
            # get output from expert
            expert_output = expert(x)
            
            # multiply by gating network's output
            # TODO
            
            # aggregate expert outputs
            # TODO 
        return x


##########----------------------------------###########
#Training the MoE model 
##########----------------------------------###########
def train_moe_mnist():
    return 

if __name__=="__main__":
    train_moe_mnist()