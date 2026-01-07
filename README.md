# Mixture of Experts Project

This repository contains our implementation and experiments with **Mixture of Experts (MoE)** models as part of the [Uppsala University Data Science Master's Project Course, 2025].

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
We had several different versions of the code from start to finish and restructured it several times so code specific to one person may be hard to find in the final version as we continuously adapted it to be more modular and coherent. Also we worked on several differnt branches and merged and combined the code base several times. In general the code distribution looked like this:  
Adam - worked on the CIFAR-10 dataset and made the Sparse and Soft MoE networks and also worked on generalization results of these models and their training, also did some visualizations  
Sourav - worked on the CIFAR-10 dataset and making MoE models to compare, also worked on some generalization of the CIFAR-10 MoE models, also worked on some visualizations  
Caleb - worked on MNIST dataset and making the Sparse and Soft MoE networks, also worked on the training of these models and producing some results, also did some visulations   
Daniel - worked on the MNIST dataset and refactoring Caleb's MoE models to work with Adam's code, also worked on eigenvalue and generalization code for MNIST, also did some visualizations, also did the final merging of the code base  
