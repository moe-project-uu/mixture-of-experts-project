import time
import numpy as np
import torch

from scripts.visualization import plot_training_curve, plot_utilization_histogram, plot_utilization_trends, plot_expert_activation
from scripts.data import load_data,create_class_dataloaders
from scripts.training import training_loop
from scripts.hard_moe import HardMOE
from scripts.eval import calculate_accuracy

def hardMoE_plots(expert_num = 4, expert_hidden = 128, topk = 2, epochs = 10, figsize = (10,6)):
    # Set parameters
    DATA_DIR = "../../data"
    FIGURE_PATH = "./HardMoE_Plots/"
    experiment_metrics = {}
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GATING_HIDDEN_SIZE = 10    # NOTE: currently this is not used by the gating network I believe TODO: look into this
    MNIST_LINEAR_LENGTH = 784  # Input
    MNIST_CLASS_NUM = 10       # Output
    #these are really the parameters that will change
    EXPERT_NUM = expert_num
    EXPERT_HIDDEN_SIZE = expert_hidden
    TOPK = topk
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EPOCHS = epochs
    model = HardMOE(
        input_size=MNIST_LINEAR_LENGTH,
        hidden_size_experts=EXPERT_HIDDEN_SIZE,
        hidden_size_gating=GATING_HIDDEN_SIZE,
        num_experts=EXPERT_NUM,
        output_size=MNIST_CLASS_NUM,
        topk=TOPK,
        device=DEVICE           # NOTE: device should not be parameter TODO: create ".to(DEVICE)" function in SoftMoE implementation? 
    )

    train_dataset, test_dataset = load_data(DATA_DIR)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset_size = test_dataset.data.shape[0]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_dataset_size, shuffle=True)

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
    print(f"Training took {training_duration} seconds")
    
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

    ## --- Stuff to look at for plotting -- ##

    plot_training_curve(
        title = "Training curves", 
        train_costs = training_loss, 
        test_costs = test_loss,
        train_accuracy = training_accuracy,
        test_accuracy = test_accuracy,
        batch_size = BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        training_time = training_duration,
        epochs = EPOCHS,
        save_path = FIGURE_PATH + f"Training_curve_{EXPERT_NUM}_experts_{EXPERT_HIDDEN_SIZE}_hidden size_{TOPK}_topk.png"
    )
    print("Plotted training curve")
    epochs_to_plot = [round(i * (len(test_loss)-4) / 4) for i in range(5)]

    plot_utilization_histogram(
        epochs_to_plot=epochs_to_plot,
        utilization_data= expert_utilization_history,
        save_path = FIGURE_PATH + f"Utilization_hist_{EXPERT_NUM}_experts_{EXPERT_HIDDEN_SIZE}_hidden size_{TOPK}_topk.png"
    )

    print("plotted utilization histogram")

    plot_utilization_trends(
        utilization_data = expert_utilization_history,
        save_path=FIGURE_PATH + f"Utilization_Trends_{EXPERT_NUM}_experts_{EXPERT_HIDDEN_SIZE}_hidden size_{TOPK}_topk.png"
    )
    print("plotted utilization trends")

    #we want to look at the expert activations at the end of training and on the test set
    class_num = 10
    test_loaders = create_class_dataloaders(test_dataset, class_num=class_num)
    train_loaders = create_class_dataloaders(train_dataset, class_num=class_num)
    # Initialize lists to store a single value (the statistic) for each digit (0-9)
    expert_probs_per_digit_train = [None] * class_num
    expert_probs_per_digit_test = [None] * class_num

    #evaluate model on the training and test set for each unique digit
    model.eval()

    #entire training set
    for idx, loader in enumerate(train_loaders):
            with torch.no_grad():
                # Since each loader contains ALL samples for one digit, we expect only ONE batch.
                for data, label in loader:
                    data = data.to(DEVICE)
                    label = label.to(DEVICE)

                    # Full model forward pass
                    _, gating_outputs, _ = model(data)
                    
                    # Calculate Mean Expert Probabilities for this digit
                    # The result is the average probability for each expert across all samples
                    mean_expert_probs = torch.mean(gating_outputs.detach().cpu(), dim=0).tolist()
                    
                    expert_probs_per_digit_train[idx] = mean_expert_probs
                    
                    break # Only one batch is expected
    expert_probs_per_digit_train = np.array(expert_probs_per_digit_train)

    #plot the train set expert activations
    plot_expert_activation(
        expert_probabilties = expert_probs_per_digit_train.T,
        expert_num = EXPERT_NUM,
        figsize=figsize,
        save_path = FIGURE_PATH + f"Expert_Activations_trainset_{EXPERT_NUM}_experts_{EXPERT_HIDDEN_SIZE}_hidden size_{TOPK}_topk.png"
    )

    #entire test set
    for idx, loader in enumerate(test_loaders):
        with torch.no_grad():
            # Since each loader contains ALL samples for one digit, we expect only ONE batch.
            for data, label in loader:
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                # Full model forward pass
                _, gating_outputs, _ = model(data)
                
                # Calculate Mean Expert Probabilities for this digit
                # The result is the average probability for each expert across all samples
                mean_expert_probs = torch.mean(gating_outputs.detach().cpu(), dim=0).tolist()
                
                expert_probs_per_digit_test[idx] = mean_expert_probs
                
                break # Only one batch is expected
    expert_probs_per_digit_test = np.array(expert_probs_per_digit_test)

    #plot the test set expert activations
    plot_expert_activation(
        expert_probabilties = expert_probs_per_digit_test.T,
        expert_num = EXPERT_NUM,
        figsize=figsize,
        save_path = FIGURE_PATH + f"Expert_Activations_testset_{EXPERT_NUM}_experts_{EXPERT_HIDDEN_SIZE}_hidden size_{TOPK}_topk.png"
    )

    print("plotted expert activation heat map")

if __name__ == "__main__":
    hardMoE_plots(expert_num=2, expert_hidden=512, topk=1, epochs = 500)
    hardMoE_plots(expert_num=4, expert_hidden=128, topk = 1, epochs = 500)
    hardMoE_plots(expert_num=8, expert_hidden=256, topk = 2, epochs = 500)
    hardMoE_plots(expert_num=16, expert_hidden=512, topk= 2, figsize=(10,4), epochs=100)
