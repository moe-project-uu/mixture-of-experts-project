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
        batch_loss = []
        batch_accuracy = []
        epoch_gating_outputs = [] 
        
        # Loop through batches (Training Phase)
        for _, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            
            # Forward pass
            outputs, gating_output, loss_importance = model(data)
            epoch_gating_outputs.append(gating_output.detach().cpu())
            
            # Evaluate
            loss = loss_function(outputs, label) + loss_importance 
            batch_loss.append(loss.item())
            batch_accuracy.append(calculate_accuracy(outputs.detach(), label.detach()))
            
            # Backward pass setting gradients to zero
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Combine all gating outputs from the epoch into a single tensor
        all_gating_outputs = torch.cat(epoch_gating_outputs, dim=0) # list of (B, N) tensors -> (Total Samples, N)
        #print("All gating outputs sumed", all_gating_outputs.sum(dim = 0))
        # Calculate the average utilization for each expert over the entire training set
        avg_expert_utilization = all_gating_outputs.mean(dim=0).numpy() # (Total Samples, N) -> (N,)
                
        # Aggregate batch matrics
        training_accuracy.append(np.average(batch_accuracy))
        training_loss.append(np.average(batch_loss))
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
                importance_loss_history.append(importance_loss)
                
                # Use .detach() for metric calculation
                test_accuracy_list.append(calculate_accuracy(test_predictions.detach(), label.detach())) 

        # Aggregate batch matrics
        test_loss.append(np.average(test_loss_list))
        test_accuracy.append(np.average(test_accuracy_list))
        if epoch % print_freq == 0:
            print(f"Epoch: {epoch} done. Test loss {test_loss[-1]:.4f}. Test accuracy {test_accuracy[-1]:.4f}")
    
    return training_loss, training_accuracy, test_loss, test_accuracy, expert_utilization_history