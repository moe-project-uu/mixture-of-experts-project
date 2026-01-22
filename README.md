# Mixture of Experts Project

This repository contains our implementation and experiments with **Mixture of Experts (MoE)** models as part of the [Uppsala University Data Science Master's Project Course, 2025].

## arXiv link: 
https://arxiv.org/abs/2601.15021

## Setup

- **Python 3.11+**

1. Create a virtual environment:

   `python -m venv .venv`

2. Activate it:

   `source .venv/bin/activate`

3. Install dependencies:

   `pip install -r requirements.txt`

4. Install the local package:

   `pip install -e .`

5. To train a model on MNIST dataset (types can be "Dense", "SoftMoE" and "SparseMoE"):

   ```bash
   python scripts/train_mnist.py \
   --FF_layer Dense \
   --epochs 50
   ```

## Project Structure

Important directories and files:

- **checkpoints:** Already trained models from different experiments.
- **notebooks**: Most of the experiments were done using jupyter notebook that can be found in this directory.
- **scripts**:
  - _train_cifar.py_ and _train_mnist.py_ are the model training scripts on the CIFAR-10 and MNIST datasets
  - _cifar10_hessian.py_ is the script that calculate the hessian on the CIFAR-10 dataset
- **src:** MoE implementation, data processing scripts and visualization scripts


## Contributions  
At an early stage, the project was split into two subgroups: CIFAR-10 (Adam, Sourav) and MNIST (Caleb, Daniel). Caleb and Daniel implemented the Dense baseline and Mixture-of-Experts (MoE) variants for MNIST, while Adam implemented the corresponding models for CIFAR-10. Sourav implemented early CIFAR-10 MoE variants and supporting experiments on separate branches; these were used for exploratory comparison during development but are not included in the final consolidated main branch.

For generalization analysis, Daniel implemented MNIST generalization experiments, while Adam implemented CIFAR-10 generalization experiments. Toward the end of the project, the goal was to produce a single consistent and modular codebase. This required substantial refactoring and consolidation, performed primarily by Daniel, who merged the MNIST and CIFAR-10 pipelines and removed duplicate or intermediate implementations. As a result, much of the final code resides in files originally written by Adam, although the design reflects iterative collaboration and multiple development stages.

In general, the contribution distribution was as follows:

Adam – Implemented Sparse, Soft, and Hard MoE networks for CIFAR-10, including load-balancing strategies and true conditional computation. Implemented CIFAR-10 generalization analysis using PyHessian and produced corresponding visualizations.

Sourav – Implemented early CIFAR-10 MoE variants and exploratory generalization experiments and visualizations used for comparison during development.

Caleb – Implemented MNIST Sparse and Soft MoE models, trained these models, and produced experimental results and visualizations.

Daniel – Implemented MNIST Sparse and Soft MoE models, trained these models, and produced experimental results and visualizations. Refactored MNIST MoE implementations to integrate with the CIFAR-10 codebase, implemented MNIST eigenvalue and generalization analysis, produced visualizations, and performed the final merging and restructuring of the codebase.
