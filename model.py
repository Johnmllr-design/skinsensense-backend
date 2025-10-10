import torch.nn as nn
from torchvision import transforms as TF
from torch import Tensor
from PIL import Image
import torch.nn.functional as F
to_image = TF.ToPILImage()

class skin_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        # ensure that the image size matched the required model dimensions
        self.get_input = TF.Compose([TF.Resize((480, 480)), TF.ToTensor()])

        # convolutional and pooling layers
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.second_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.third_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fourth_conv = nn.Conv2d(in_channels=128, out_channels=200, kernel_size= 3, padding=1)
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layers
        self.FC_layer_1 = nn.Linear(in_features= 180000, out_features= 1000, bias=True)
        self.FC_layer_2 = nn.Linear(in_features=1000, out_features= 100, bias=True)
        self.FC_layer_3 = nn.Linear(in_features=100, out_features=1)
        self.flattener = nn.Flatten()

    

    def forward(self, input_image_file):

        # turn the image into a tensor
        tensor:Tensor = self.get_input(input_image_file)

        # Unsqueeze for a batch layer
        tensor = tensor.unsqueeze(0)

        # convolve and pool
        first_conv = F.relu(self.initial_conv(tensor))
        pooled = self.pool_layer(first_conv)
        second_conv = F.relu(self.second_conv(pooled))
        pooled = self.pool_layer(second_conv)
        third_conv = F.relu(self.third_conv(pooled))
        pooled = self.pool_layer(third_conv)
        fourth_conv = F.relu(self.fourth_conv(pooled))
        pooled = self.pool_layer(fourth_conv)

        #flatten and pass through fully connected layers
        flattened_tensor = self.flattener(pooled)
        connected_output_1 = F.relu(self.FC_layer_1(flattened_tensor))
        connected_output_2 = F.relu(self.FC_layer_2(connected_output_1))
        connected_output_3 = self.FC_layer_3(connected_output_2)

        # return the output
        return connected_output_3
    



