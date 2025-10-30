import torch

def calculate_accuracy(outputs, labels):
    _, output_index = torch.max(outputs, 1)
    return (output_index == labels).sum().item() / labels.size(0)

def eval_loop(
    loader, 
    model, 
    model_gating,
    loss_function,
    device
):
    """Evaluates the model and its gating mechanism on a single batch of data.

    This function calculates loss, accuracy, and the mean expert probabilities 
    for the batch. It's typically used to evaluate a specific data subset.

    Parameters:
        loader: The data loader, expected to contain a single batch of samples.
        model: The main neural network (MoE) model for making predictions.
        model_gating: The gating network component that assigns probabilities to experts.
        loss_function: The criterion used to calculate the prediction loss.
        device: The device ('cpu' or 'cuda') for computation.

    Returns:
        A dictionary with the evaluation metrics:
            - 'loss': The calculated loss for the batch.
            - 'accuracy': The calculated accuracy for the batch.
            - 'mean_expert_probs': The average probability assigned to each expert.
    """
    metrics = {}
    
    with torch.no_grad():
        # Since each loader contains ALL samples for one digit, we expect only ONE batch.
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            
            # Gating layer forward pass (assumes moe_gating returns a dict with "probs")
            test_probs_output = model_gating(data) 
            
            # Full model forward pass
            test_predictions, _, _ = model(data)
            
            # Calculate Mean Expert Probabilities for this digit
            # The result is the average probability for each expert across all samples
            mean_expert_probs = torch.mean(test_probs_output["probs"], dim=0).tolist()
            
            batch_loss = loss_function(test_predictions, label)
            batch_accuracy = calculate_accuracy(test_predictions.data, label)
            
            metrics["loss"] = batch_loss.item()
            metrics["accuracy"] = batch_accuracy
            metrics["mean_expert_probs"] = mean_expert_probs
            
            break # Only one batch is expected
    
    return metrics