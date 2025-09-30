import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm
from typing import Tuple

# Import our custom utilities
from utils import get_dataloaders, calculate_model_stats, CSVLogger

# --- Training and Evaluation Functions ---

def run_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion, optimizer, device: torch.device, is_training: bool = True) -> Tuple[float, float]:
    """Runs one epoch of training or evaluation."""
    model.train() if is_training else model.eval()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    desc = "Training" if is_training else "Evaluating"
    context = torch.enable_grad() if is_training else torch.no_grad()
    
    with context:
        progress_bar = tqdm(loader, desc=desc, leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{total_loss/total_samples:.4f}", 
                acc=f"{correct_predictions/total_samples:.4f}"
            )
    
    return total_loss / total_samples, correct_predictions / total_samples

def save_checkpoint(model: nn.Module, optimizer, epoch: int, best_acc: float, filepath: str):
    """Saves model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(state, filepath)
    print(f"\nCheckpoint saved to {filepath}")

def setup_model(model_name: str, num_classes: int) -> nn.Module:
    """Load pre-trained model and modify for specified number of classes."""
    print(f"Loading pre-trained model: {model_name}")
    model = models.get_model(model_name, weights='IMAGENET1K_V1')
    
    # Modify the final layer for the target number of classes
    if 'resnet' in model_name or 'resnext' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif 'vgg' in model_name:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        raise NotImplementedError(f"Model architecture {model_name} not supported for classifier modification.")
    
    return model

def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Training model: {args.model_name}")

    # Initialize CSV logger for tracking metrics
    log_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_training_log.csv")
    logger = CSVLogger(log_path, ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'])
    print(f"Logging metrics to: {log_path}")

    # Data
    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Model
    model = setup_model(args.model_name, num_classes=200).to(device)
    
    # Calculate and display model stats
    calculate_model_stats(model)

    # Optimizer, Loss, and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training Loop
    best_acc = 0.0
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer, device, is_training=False)
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Log metrics to CSV
        metrics = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 6),
            'learning_rate': current_lr
        }
        logger.log(metrics)
        
        print(f"Epoch {epoch} Summary: ")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {val_loss:.4f}  | Test Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)
    
    print("\n--- Training Complete ---")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Best model saved at: {checkpoint_path}")
    print(f"Training metrics logged to: {log_path}")
    print(f"\nTo plot training graphs, run:")
    print(f"  python -m utils.plot_metrics --csv-path {log_path} --save-dir {args.checkpoint_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN Baseline on CUB-200-2011")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the CUB_200_2011 dataset directory")
    parser.add_argument('--model-name', type=str, default='resnet50', help="Name of the torchvision CNN model to use (e.g., 'resnet50', 'vgg16')")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help="Directory to save model checkpoints")
    
    args = parser.parse_args()
    main(args)