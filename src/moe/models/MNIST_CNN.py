import torch

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