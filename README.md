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
