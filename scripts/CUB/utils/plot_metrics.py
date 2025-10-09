import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_training_metrics(csv_path: str, save_dir: Optional[str] = None):
    """
    Plot training metrics from CSV log file.
    
    Args:
        csv_path: Path to the CSV log file
        save_dir: Directory to save plots (optional, if None plots are only displayed)
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot 1: Loss over epochs
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy over epochs
    ax2.plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate over epochs
    ax3.plot(df['epoch'], df['learning_rate'], label='Learning Rate', marker='d', color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for better visualization
    
    # Plot 4: Validation accuracy with best epoch marked
    ax4.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s', color='orange')
    best_idx = df['val_acc'].idxmax()
    best_epoch = df.loc[best_idx, 'epoch']
    best_acc = df.loc[best_idx, 'val_acc']
    ax4.scatter([best_epoch], [best_acc], color='red', s=100, zorder=5, label=f'Best: {best_acc:.4f} @ Epoch {best_epoch}')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy')
    ax4.set_title('Validation Accuracy (with Best Epoch)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Best Validation Accuracy: {best_acc:.4f} at Epoch {best_epoch}")
    print(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final Train Accuracy: {df['train_acc'].iloc[-1]:.4f}")
    print(f"Final Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"Final Validation Accuracy: {df['val_acc'].iloc[-1]:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV log")
    parser.add_argument('--csv-path', type=str, required=True, help="Path to the CSV log file")
    parser.add_argument('--save-dir', type=str, help="Directory to save plots")
    
    args = parser.parse_args()
    plot_training_metrics(args.csv_path, args.save_dir)