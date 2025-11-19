import torch.nn as nn;
import torch.nn.functional as F;

#Convolution Output Size Formula / 2*pooling_size

#CNN Class
class CNN_Classifier(nn.Module):
    #Setup for parameters
    def __init__(self, out_channels1 = 8, out_channels2 = 16, conv_stride = 1, pool_stride = 2, pooling_size = 2, padding = 1, kernel_size = 3):
        super(CNN_Classifier, self).__init__()
        final_size = (int)((((28 - kernel_size + (2*pooling_size)) / conv_stride) + 1) / (2*pooling_size))
        #First Convolution Layer ()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels1, stride=conv_stride, kernel_size=kernel_size, padding=padding)
        #Second Convolution Layer
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, stride=conv_stride, kernel_size=kernel_size, padding=padding)
        #Pooling Layer (2 by 2)
        self.pool = nn.MaxPool2d(pooling_size, pool_stride)
        #Fully Connected Layer (16 * 14 * 14 is the final shape after pooling)
        self.fc1 = nn.Linear(out_channels2 * final_size * final_size, 10) #

    def forward(self, data):
        #Convolution Layer 1
        data = F.relu(self.conv1(data))
        #Pooling 1
        data = self.pool(data)
        #Convolution Layer 2
        data = F.relu(self.conv2(data))
        #Pooling 2
        data = self.pool(data)

        #Flatten the data
        data = data.view(data.size(0), -1)
        data = self.fc1(data)

        return data


# images, labels = next(iter(train_data))
# activation_layers = model(images)
# print(activation_layers.shape)

