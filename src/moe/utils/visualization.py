import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

def plot_training_curve(
    title: str, 
    train_costs: List[float], 
    test_costs: List[float], 
    train_accuracy: List[float], 
    test_accuracy: List[float], 
    batch_size: int, 
    learning_rate: float, 
    training_time: float, 
    epochs: int,
    save_path: Optional[str] = None
) -> None:
    """Create a figure with two subplots showing training metrics over epochs.
    
    Displays training/testing costs and accuracies side by side, with a title showing
    hyperparameters and training duration.

    Parameters:
        title: Title of the figure.
        train_costs: List of training costs for each epoch.
        test_costs: List of testing costs for each epoch.
        train_accuracy: List of training accuracy values for each epoch.
        test_accuracy: List of testing accuracy values for each epoch.
        batch_size: Size of training batches used.
        learning_rate: Learning rate used during training.
        training_time: Total training duration in seconds.
        epochs: Total number of training epochs.
        save_path: Optional path to save the figure.
    """
    lg=18
    md=13
    sm=9
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.15, fontsize=lg)
    
    ## Add subtitle with hyperparameters and training time
    sub = f'| Batch size:{batch_size} | Learning rate:{learning_rate} | Number of Epochs:{epochs} | Training Time:{round(training_time)}sec |'
    fig.text(0.5, 0.99, sub, ha='center', fontsize=md)
    
    x = range(1, len(train_costs)+1)
    
    ## Cost/Loss subplot
    axs[0].plot(x, train_costs, label=f'Final train cost: {train_costs[-1]:.4f}')
    axs[0].plot(x, test_costs, label=f'Final test cost: {test_costs[-1]:.4f}')
    axs[0].set_title('Costs', fontsize=md)
    axs[0].set_xlabel('Epochs', fontsize=md)
    axs[0].set_ylabel('Cost', fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis='both', labelsize=sm)
    
    ## Accuracy subplot
    axs[1].plot(x, train_accuracy, label=f'Final train accuracy: {100*train_accuracy[-1]:.2f}%')
    axs[1].plot(x, test_accuracy, label=f'Final test accuracy: {100*test_accuracy[-1]:.2f}%')
    axs[1].set_title('Accuracy', fontsize=md)
    axs[1].set_xlabel('Epochs', fontsize=md)
    axs[1].set_ylabel('Accuracy (%)', fontsize=sm)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis='both', labelsize=sm)
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)
    
def plot_utilization_histogram(
    epochs_to_plot: List[int],
    utilization_data: List[np.ndarray],
    expert_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """Plot expert utilization as a series of bar charts across specified epochs.
    
    Creates a horizontal series of bar plots, each showing the utilization distribution
    of experts for a given epoch. Utilization values are displayed on top of each bar.

    Parameters:
        epochs_to_plot: List of epoch numbers to visualize.
        utilization_data: List of numpy arrays of expert utilizations.
        expert_names: Optional list of names for each expert. Defaults to 'Expert i'.
        figsize: Tuple of (width, height) for the figure size.
        save_path: Optional path to save the figure.
    """
    num_epochs = len(epochs_to_plot)
    max_epoch_requested = max(epochs_to_plot) 

    if num_epochs == 0:
        print("No epochs to plot.")
        return
    
    if len(utilization_data) <= max_epoch_requested:
        raise ValueError(f"Utilization data length ({len(utilization_data)}) is less than or equal to the maximum epoch requested ({max_epoch_requested}).")

    # Use a single row layout for simplicity
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_epochs,
        figsize=(figsize[0], figsize[1]),
        sharey=True # Share the y-axis (0-1 range)
    )

    # If there is only one subplot, axes is not an array, so make it one
    if num_epochs == 1:
        axes = [axes]

    # Determine expert names if not provided
    first_epoch_data = utilization_data[epochs_to_plot[0]]
    num_experts = len(first_epoch_data)
    if expert_names is None:
        expert_names = [f'Expert {i}' for i in range(num_experts)]

    x = np.arange(num_experts) # the label locations
    width = 0.8 # the width of the bars

    for ax, epoch in zip(axes, epochs_to_plot):
        utilization = utilization_data[epoch]

        if utilization is None:
            print(f"Warning: No data found for epoch {epoch}. Skipping.")
            continue

        # Create the bar plot
        ax.bar(x, utilization, width, color='skyblue', edgecolor='black')

        # Set title and labels
        ax.set_title(f'Epoch {epoch}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(expert_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0) # Set the y-limit from 0 to 1 as requested
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Only set y-label on the first subplot
        if ax == axes[0]:
            ax.set_ylabel('Average Utilization (0 to 1)', fontsize=10)

        # Add utilization values on top of the bars
        for i, val in enumerate(utilization):
            ax.text(x[i], val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.suptitle('Expert Utilization Across Epochs', y=1.02, fontsize=14, fontweight='bold')
    
    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)
    
def plot_utilization_trends(
    utilization_data: List[np.ndarray],
    expert_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """Visualize expert utilization trends over time using a line plot.
    
    Creates a single plot showing how each expert's utilization changes across epochs,
    with different colored lines for each expert and markers at data points.

    Parameters:
        utilization_data: List of numpy arrays of expert utilizations. Each element
                          in the list represents a single epoch.
        expert_names: Optional list of names for each expert. Defaults to 'Expert i'.
        figsize: Tuple of (width, height) for the figure size.
        save_path: Optional path to save the figure.
    """
    if not utilization_data:
        print("Utilization data is empty. Nothing to plot.")
        return

    # Stack them vertically (rows are epochs, columns are experts)
    # The utilization_data is a list where each element is the utilization array for one epoch.
    utilization_matrix = np.vstack(utilization_data)
    
    # Define the epochs corresponding to the data points
    # Since utilization_data is a list of utilization arrays, its length is the number of epochs.
    num_epochs = len(utilization_data)
    # Epoch numbers start from 0 up to num_epochs - 1
    epochs = np.arange(num_epochs)

    # Transpose to get (rows are experts, columns are epochs)
    expert_utilization_trends = utilization_matrix.T

    num_experts = expert_utilization_trends.shape[0]

    # Determine expert names if not provided
    if expert_names is None:
        expert_names = [f'Expert {i}' for i in range(num_experts)]
    elif len(expert_names) != num_experts:
        print(f"Warning: Expert names list length ({len(expert_names)}) does not match number of experts ({num_experts}). Using default names.")
        expert_names = [f'Expert {i}' for i in range(num_experts)]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(num_experts):
        expert_data = expert_utilization_trends[i]
        expert_name = expert_names[i]

        # Plot line and scatter points. The x-axis is now correctly set to 'epochs'.
        if num_epochs > 100:
            # For large number of epochs, use a line plot without markers for clarity
            ax.plot(epochs, expert_data, label=expert_name, linestyle='-')
        else:
            ax.plot(epochs, expert_data, label=expert_name, marker='o', linestyle='-')

    ax.set_title('Expert Utilization Trends Across Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Utilization (0 to 1)', fontsize=12)

    # Set x-ticks to be only the plotted epoch numbers
    ax.set_xticks(epochs)

    # Ensure y-axis is from 0 to 1
    ax.set_ylim(0, 1.0)

    # Add legend
    ax.legend(title="Experts", loc='best', fontsize=10)

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)
    

def plot_expert_activation(
    expert_probabilties: List[float],
    expert_num: int,
    save_path: Optional[str] = None
) -> None:
    """
        Plots the activation probabilities of a set of experts across different classes.

        Parameters:
            expert_probabilties: A list or 2D array-like structure (e.g., numpy array) 
                                where each row corresponds to an expert and each 
                                column corresponds to a class or sample. The values 
                                represent the activation probability.
            expert_num: The total number of experts in the model, used to set the 
                        y-axis ticks and labels.
            save_path: Optional path to save the figure.
    """
    fig = plt.figure(figsize=(10, 2))
    plt.imshow(expert_probabilties, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')

    plt.title("Probabaility of each expert activating for digits in MNIST")
    plt.xlabel("Number")
    plt.ylabel("Expert")
    
    # Add x axis ticks for class labels
    plt.xticks(ticks=np.arange(10), labels=[str(i) for i in range(0, 10)])

    # Add y axis ticks for expert labels
    yticks = [i for i in range(expert_num)]
    yticks_labels = [f"Expert {i}" for i in range(expert_num)]
    plt.yticks(yticks, yticks_labels)
    
    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)