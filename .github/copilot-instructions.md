# Copilot Instructions for Mixture of Experts Project

## Project Overview
- This repository explores Mixture of Experts (MoE) architectures, focusing on PyTorch implementations for MNIST, CIFAR-10, CIFAR-100, and CUB datasets.
- Main code is organized by dataset in `scripts/` and by model architecture in `src/moe/`.
- Notebooks in `notebooks/` are used for prototyping, debugging, and visualizing results.

## Key Components
- `src/moe/`: Core MoE model implementations (e.g., `moe_mnist.py`).
- `scripts/MNIST/`, `scripts/CUB/`: Training scripts and utilities for each dataset.
- `notebooks/`: Contains baseline experiments and model testing (e.g., `MNIST_testing_notebook.ipynb`).
- `data/`: Organized by dataset, with images in subfolders for each class.

## Developer Workflows
- **Training**: Run scripts in `scripts/{DATASET}/train_{dataset}.py` (e.g., `train_mnist.py`).
- **Testing/Debugging**: Use notebooks in `notebooks/` for interactive experiments.
- **Data Loading**: Use provided loaders (e.g., `scripts/MNIST/load_mnist.py`).
- **Reproducibility**: Set `DEVICE` to `cuda` if available, otherwise `cpu`.
- **Checkpoints**: Models are saved in `scripts/{DATASET}/checkpoints/`.

## Patterns & Conventions
- Experts and gating networks are implemented as PyTorch `nn.Module` subclasses.
- SoftMoE pattern: All experts are evaluated, outputs are weighted by gating mechanism (see `MNIST_testing_notebook.ipynb`).
- Data is typically flattened for simple MoE models; use `linear=True` in data loaders.
- Training loops log loss and accuracy per epoch; see notebook for detailed logging.
- Use `torch.utils.data.DataLoader` for batching.

## Integration Points
- External dependencies: PyTorch, NumPy, Matplotlib, tqdm.
- Data is loaded from local folders, not via online datasets.
- No cloud or API integration detected.

## Examples
- See `MNIST_testing_notebook.ipynb` for SoftMoE implementation and training loop.
- See `src/moe/moe_mnist.py` for reusable MoE model code.
- See `scripts/MNIST/load_mnist.py` for custom data loading.

## How to Extend
- Add new experts by subclassing `Expert` and updating MoE constructors.
- Add new datasets by mirroring the folder structure and creating new loaders/scripts.
- For new experiments, create a notebook in `notebooks/` and import relevant modules.

## Build/Test Commands
- No build system detected; run scripts directly with Python.
- Example: `python scripts/MNIST/train_mnist.py`
- For notebooks, use Jupyter or VS Code's notebook interface.

---
_If any section is unclear or missing, please provide feedback so instructions can be improved._
