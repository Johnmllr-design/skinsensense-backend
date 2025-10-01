import torch.nn as nn
import torchvision
from torch import Tensor
import torch.nn.functional as F

class skin_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.second_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.third_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FC_layer_1 = nn.Linear(in_features= (128 * 60 * 60), out_features= 1000, bias=True)
        self.FC_layer_2 = nn.Linear(in_features=1000, out_features= 100, bias=True)
        self.FC_layer_3 = nn.Linear(in_features=100, out_features=1)
        self.flattener = nn.Flatten()

    

    def forward(self, tensor):
        first_conv = F.relu(self.initial_conv(tensor))
        pooled = self.pool_layer(first_conv)
        second_conv = F.relu(self.second_conv(pooled))
        pooled = self.pool_layer(second_conv)
        third_conv = F.relu(self.third_conv(pooled))
        pooled = self.pool_layer(third_conv)
        flattened_tensor = self.flattener(pooled)
        connected_output_1 = F.relu(self.FC_layer_1(flattened_tensor))
        connected_output_2 = F.relu(self.FC_layer_2(connected_output_1))
        connected_output_3 = F.sigmoid(self.FC_layer_3(connected_output_2))
        return connected_output_3


