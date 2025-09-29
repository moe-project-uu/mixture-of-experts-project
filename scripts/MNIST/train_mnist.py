#modified from the deeplearning course at Uppsala University
"""
Train a CNN classifier for the MNIST dataset which is under data/MNIST
This is a dense baseline comparison to the MoE model
Components of the script are:
1. Data loading
2. Model definition
3. Training loop
4. Evaluation
5. Saving the model
6. Loading the model
7. Testing the model
"""

#imports
import numpy as np 
import random
import os
import sys
#adding current directory to path so we can import load_mnist
# dir = os.path.abspath(".")  
# if dir not in sys.path:
#     sys.path.append(dir)
from . import load_mnist
import time
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

##########----------------------------------###########
#Data loading
##########----------------------------------###########

def get_data():
    #get the train and test data from the dataset
    xtrain,ytrain,xtest,ytest = load_mnist.load_mnist()
    #converting to Tensors for easy PyTorch implementation and reshape for a CNN
    xtrain = torch.Tensor(xtrain).reshape(60000, 1,28,28).to(DEVICE)
    ytrain = torch.Tensor(ytrain).to(DEVICE)
    xtest = torch.Tensor(xtest).reshape(10000, 1,28,28).to(DEVICE)
    ytest = torch.Tensor(ytest).to(DEVICE)
    #first we want to put our data in a pytorch dataset so we can mini batch and enumerate through it later more easily
    train_dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
    test_dataset = torch.utils.data.TensorDataset(xtest, ytest)
    #Making a dataloader for this specific CNN which is a wrapper around the Dataset for easy use
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #make the batch size for the test DataLoader the size of the dataset for evaluation.
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = ytest.shape[0], shuffle=True)
    return train_loader, test_loader

#calculating the accuracy given outputs not softmaxed and labels one hot encoding used for evaluation during training and testing
def calculate_accuracy(outputs, labels):
    #don't need to softmax because the max value will be the max softmax we just pull the index to get the digit prediction 
    _, output_index = torch.max(outputs,1)
    #get the index/ digit of the label
    _, label_index = torch.max(labels, 1)
    # return the number of correct matches and divide by the size to get accuracy
    return (output_index == label_index).sum().item()/labels.size(0)

##########----------------------------------###########
#Model definition: 3 layered CNN network
#input is 1x28x28 
#layer 1: conv(3)-relu-pool(2,2) -> 8x14x14
#layer 2: conv(3)-relu-pool(2,2) -> 16x7x7  
#layer 3: conv(3)-relu -> 32x7x7
#output layer: linear -> 10
##########----------------------------------###########
class MNIST_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #1 input channel, 8 output channels, kernel size 3, stride 1, padding 1
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        #non linearity
        self.relu1 = torch.nn.ReLU()
        #first pooling layer with kernel size 2, stride 2 reduces image to (8,14,14)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        #8 input channels, 16 output channels, kernel size 3, stride 1 padding 1
        self.conv2 = torch.nn.Conv2d(in_channels= 8,out_channels= 16 , kernel_size= 3, stride= 1, padding= 1)
        #non linearity
        self.relu2 = torch.nn.ReLU()
        #second pooling layer with kernel size 2, stride 2 reduces image to (16,7,7)
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # 16 inputs, 32 outputs, kernel size 3, stride 1, padding 1
        self.conv3 = torch.nn.Conv2d(in_channels= 16,out_channels= 32 , kernel_size= 3, stride= 1, padding= 1)
        #non linearity
        self.relu3 = torch.nn.ReLU()
        #output netwrok we have 32 channels and an image that is (7,7)
        self.output = torch.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        #pass through the first convolution and relu and pooling layers
        x = self.pool1(self.relu1(self.conv1(x)))
        #pass through the second convolution and relu and pooling layers
        x = self.pool2(self.relu2(self.conv2(x)))
        #pass through the final convolution and relu
        x = self.relu3(self.conv3(x))
        #flatten all dimensions except batch dimension which is dimension 0 so we start at 1
        x = torch.flatten(x, 1)
        #pass through our output layer
        x = self.output(x)
        return x
    
##########----------------------------------###########
#Training loop
##########----------------------------------###########

def training_loop(train_loader, test_loader, num_epochs, model, loss_function, optimizer):
    #arrays for our plots
    training_loss = []
    training_accuracy = []
    test_loss = []
    test_accuracy =[]
    #Setting up the training loop
    print("Starting the Training Loop")
    for epoch in range(num_epochs):
        #keep the loss and accuracies after each mini batch
        batch_loss = []
        batch_accuracy = []
        #loop through a mini-batch on the same train loadear
        for batch_index, (data, label) in enumerate(train_loader):
            # Forward pass
            outputs = model(data)
            #evaluate the loss
            loss = loss_function(outputs, label)
            #append the loss to the batch loss
            batch_loss.append(loss.item())
            #calculate the accuracy based on the outputs (not softmaxed) and labels. Do outputs.data so we don't pass gradient info
            batch_accuracy.append(calculate_accuracy(outputs.data, label))

            # Backward pass setting gradients to zero
            optimizer.zero_grad()
            #calcualting gradients
            loss.backward()
            #updating parameters
            optimizer.step()

        #add to the training epoch accuracies and losses
        training_accuracy.append(np.average(batch_accuracy))
        training_loss.append(np.average(batch_loss))
        #get the test loss and accuracy
        #change mode
        model.eval()
        #so we don't accidentally change anything
        with torch.no_grad():
            #get the "batch" of the test data which is all of it
            for batch_index, (data, label) in enumerate(test_loader):
                #get our test predicitons
                test_predictions = model(data)
                #test loss and move to cpu so I can plot
                loss = loss_function(test_predictions, label).to("cpu")
                #append statistics
                test_loss.append(loss)
                test_accuracy.append(calculate_accuracy(test_predictions.data, label))
        #back to training mode
        model.train()
        #printing
        print(f"Epoch: {epoch} done. Test loss {test_loss[epoch]}. Test accuracy {test_accuracy[epoch]}")
    return training_loss, training_accuracy, test_loss, test_accuracy

##########----------------------------------###########
#Training the model
##########----------------------------------###########
def train_mnist():
    #get the data
    train_loader, test_loader = get_data()
    #set the random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    #Make the CNN neural netowrk model
    model = MNIST_CNN().to(DEVICE)
    #Our loss function will be cross entropy since we are getting a probability distribution
    loss = torch.nn.CrossEntropyLoss()
    #Here we are going to use classic stochastic gradient descent without any special optimizations since we will change this later
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)

    #find the start time
    start = time.time()

    #run the training loop
    training_loss, training_accuracy, test_loss, test_accuracy = training_loop(train_loader, test_loader, 
    EPOCHS, model, loss, optimizer)

    #end time and get the total time
    end= time.time()
    total_time = end - start

    # save final model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(
        {'model': model.state_dict(), 'test_accuracy': test_accuracy, 'test_loss': test_loss, 
        'training_accuracy': training_accuracy, 'training_loss': training_loss, 'total_time': total_time,
        'optimizer': optimizer.state_dict()},
        'checkpoints/MNIST.pt'
    )
    print(f"Saved checkpoint to checkpoints/MNIST.pt")

if __name__=="__main__":
    print(os.getcwd())
    print(sys.path)
    train_mnist()