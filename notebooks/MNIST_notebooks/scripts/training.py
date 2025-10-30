import torch
import numpy as np
from tqdm import tqdm

from .eval import calculate_accuracy

def training_loop(
    train_loader, 
    test_loader, 
    num_epochs, 
    model, 
    loss_function, 
    optimizer, 
    device,
    print_freq = 10
):
    model = model.to(device)
    
    ## Metrics
    training_loss = []
    training_accuracy = []
    test_loss = []
    test_accuracy = []
    expert_utilization_history = []
    importance_loss_history = []
    
    print("Starting the Training Loop")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_samples = 0
        epoch_gating_outputs = []
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and epoch % 10 == 0:
            torch.cuda.empty_cache()
        
        # Loop through batches (Training Phase)
        for _, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            batch_size = data.size(0)
            
            # Forward pass
            outputs, gating_output, loss_importance = model(data)
            epoch_gating_outputs.append(gating_output.detach())  # Keep on GPU if possible
            
            # Evaluate
            loss = loss_function(outputs, label) + loss_importance 
            epoch_loss += loss.item() * batch_size
            epoch_accuracy += calculate_accuracy(outputs.detach(), label.detach()) * batch_size
            total_samples += batch_size
            
            # Backward pass setting gradients to zero
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate epoch metrics
        epoch_loss /= total_samples
        epoch_accuracy /= total_samples
        training_loss.append(epoch_loss)
        training_accuracy.append(epoch_accuracy)

        # Calculate expert utilization
        all_gating_outputs = torch.cat(epoch_gating_outputs, dim=0)
        avg_expert_utilization = all_gating_outputs.mean(dim=0).cpu().numpy()
        expert_utilization_history.append(avg_expert_utilization)

        # Test Phase
        model.eval()
        test_loss_list = []
        test_accuracy_list = []
        
        with torch.no_grad():
            for _, (data, label) in enumerate(test_loader):
                data = data.to(device)
                label = label.to(device)
                
                # Forward pass
                test_predictions, _, importance_loss = model(data)
                
                # Evaluate
                loss = loss_function(test_predictions, label) 
                test_loss_list.append(loss.item())
                importance_loss_history.append(importance_loss.item())
                
                # Use .detach() for metric calculation
                test_accuracy_list.append(calculate_accuracy(test_predictions.detach(), label.detach()))

        # Aggregate batch metrics
        test_loss.append(np.average(test_loss_list))
        test_accuracy.append(np.average(test_accuracy_list))
        
        if (epoch+1) % print_freq == 0:
            print(f"Epoch: {epoch+1} done. Train loss: {epoch_loss:.4f}, acc: {epoch_accuracy:.4f}. Test loss: {test_loss[-1]:.4f}, acc: {test_accuracy[-1]:.4f}")
            
        # Check if model is already overfitted on the training data
        if epoch_accuracy == 1.0:
            print("Overfitted model, finishing training!")
            break
    
    return training_loss, training_accuracy, test_loss, test_accuracy, expert_utilization_history