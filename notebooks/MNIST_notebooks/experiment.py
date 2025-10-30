## Imports
import time

import numpy as np 
import pandas as pd
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from scripts import visualization
from scripts.data import load_data, create_class_dataloaders
from scripts.eval import eval_loop
from scripts.training import training_loop
from scripts.soft_moe import SoftMOE
from scripts.hard_moe import HardMOE

def run_experiment(model, epochs):
    ## Set-up
    DATA_DIR = "../../data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_metrics = {}
    
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EPOCHS = epochs

    ## Load data
    train_dataset, test_dataset = load_data(DATA_DIR)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    test_dataset_size = test_dataset.data.shape[0]
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=test_dataset_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    ## --- Training --- ###
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    training_start = time.time()
    training_loss, training_accuracy, test_loss, test_accuracy, expert_utilization_history = \
        training_loop(
            train_loader=train_loader, 
            test_loader=test_loader, 
            num_epochs=EPOCHS, 
            model=model, 
            loss_function=loss_function, 
            optimizer=optimizer,
            print_freq=50,
            device=DEVICE)
    training_end = time.time()
    training_duration = training_end-training_start
    
    ## --- Main Metrics --- ##
    experiment_metrics = {
        "M_A": np.max(training_accuracy),
        "ETT_M_A": np.argmax(training_accuracy)+1,
        "G_A": np.max(test_accuracy),
        "ETT_G_A": np.argmax(test_accuracy)+1,
        "G_A_min_loss": test_accuracy[test_loss.index(min(test_loss))],
        "ETT_G_A_min_loss": test_loss.index(min(test_loss))+1
    }
    print(f"M_A (memorization accuracy): {experiment_metrics['M_A']:.2%}")
    print(f"ETT to reach M_A: {experiment_metrics['ETT_M_A']}")
    print(f"G_A (generalization accuracy): {experiment_metrics['G_A']:.2%}")
    print(f"ETT to reach G_A: {experiment_metrics['ETT_G_A']}")
    print(f"G_A_min_loss (generalization accuracy): {experiment_metrics['G_A_min_loss']:.2%}")
    print(f"ETT to reach G_A_min_loss: {experiment_metrics['ETT_G_A_min_loss']}")
        
    return experiment_metrics

def cartesian_product(parameters, excluder_function=None):
    # Create include config list
    parameter_names = parameters.keys()
    parameter_values = parameters.values()
    config_list = [
        dict(zip(parameter_names, v)) for v in product(*parameter_values)
    ]
    
    # Filter the config list based on the excluder function
    if excluder_function is not None:
        filtered_config_list = []
        for conf in config_list:
            excluded = excluder_function(conf)
            if not excluded:
                filtered_config_list.append(conf)
        return filtered_config_list

    return config_list

def get_softmoe_parameter_sets():
    experiment_parameters = {
        "experts": [2, 4, 8, 16],
        "expert_hidden_size": [64, 128, 256, 512],
    }
    excluder_function = None
    return experiment_parameters, excluder_function

def get_hardmoe_parameter_sets():
    experiment_parameters = {
        "experts": [1, 2, 4, 8, 16],
        "expert_hidden_size": [64, 128, 256, 512],
        "topk": [1, 2, 4]
    }
    
    def hardmoe_excluder_function(conf):
        exclude = False
        exclude = exclude or conf["topk"] > conf["experts"]
        exclude = exclude or conf["topk"] == conf["experts"]
        return exclude
    
    excluder_function = hardmoe_excluder_function
    return experiment_parameters, excluder_function

def save_experiment_data(all_experiment_data):
    # Save dataframe of experimentation data
    df = pd.DataFrame(all_experiment_data) 
    experiment_data_path = f"./experiment.csv"
    df.to_csv(experiment_data_path)

if __name__ == "__main__":
    SKIP_EXPERIMENTS = 0
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 1
    TYPE = "HARDMOE" # SOFTMOE
    
    experiment_parameters = {}
    excluded_experiment_parameters = {}
    if TYPE == "HARDMOE":
        experiment_parameters, excluder_function = get_hardmoe_parameter_sets()
    elif TYPE == "SOFTMOE":
        experiment_parameters, excluder_function = get_softmoe_parameter_sets()
    
    config_list = cartesian_product(experiment_parameters, excluder_function)
    print(f"Created {len(config_list)} number of experiment configurations!")
    
    all_experiment_data = []
    for i, config in enumerate(config_list):
        if i < SKIP_EXPERIMENTS:
            print(f">>> Skipping Experiment {i:3} <<<")
            continue
        print(f">>> Starting Experiment {i:3} <<<")
        experiment_data = {}
        experiment_data.update(config) # add config
        
        ## Set generic parameters
        GATING_HIDDEN_SIZE = 10    # NOTE: currently this is not used by the gating network I believe TODO: look into this
        MNIST_LINEAR_LENGTH = 784  # Input
        MNIST_CLASS_NUM = 10       # Output
        
        ## Set configuration parameters for the experiment
        EXPERT_NUM = config["experts"]
        EXPERT_HIDDEN_SIZE = config["expert_hidden_size"]
        
        print(f"Experts: {EXPERT_NUM}")
        print(f"Expert hidden size: {EXPERT_HIDDEN_SIZE}")
        
        model = None
        ## SoftMoE
        if TYPE == "SOFTMOE":
            model = SoftMOE(
                input_size=MNIST_LINEAR_LENGTH,
                hidden_size_experts=EXPERT_HIDDEN_SIZE,
                hidden_size_gating=GATING_HIDDEN_SIZE,
                num_experts=EXPERT_NUM,
                output_size=MNIST_CLASS_NUM,
                device=DEVICE          # NOTE: device should not be parameter TODO: create ".to(DEVICE)" function in SoftMoE implementation? 
            )
        elif TYPE == "HARDMOE":
            ## HardMoE
            TOPK = config["topk"]
            print(f"Top-k: {TOPK}")
            model = HardMOE(
                input_size=MNIST_LINEAR_LENGTH,
                hidden_size_experts=EXPERT_HIDDEN_SIZE,
                hidden_size_gating=GATING_HIDDEN_SIZE,
                num_experts=EXPERT_NUM,
                output_size=MNIST_CLASS_NUM,
                topk=TOPK,
                device=DEVICE           # NOTE: device should not be parameter TODO: create ".to(DEVICE)" function in SoftMoE implementation? 
            )
        
        experiment_metrics = run_experiment(model=model, epochs=EPOCHS)
        
        experiment_data.update(experiment_metrics) # add metrics
        all_experiment_data.append(experiment_data)
        
        save_experiment_data(all_experiment_data)
        print(40*"-")
        print()