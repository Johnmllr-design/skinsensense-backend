import torch.nn as nn
import os
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision import transforms
from model import skin_cnn
from load_data import load_dataset
from torch.optim import Adam
import torch

def test():
    # hyperparameters
    model = skin_cnn()
    model.load_state_dict(torch.load("saved_models/skin_model_epoch_7.pth"))
    inputs, ground_truth = load_dataset()
    inputs = inputs[0:100]
    ground_truth = ground_truth[0:100]
    num_obs = len(inputs)
    correct = 0

    for i in range(0, num_obs):
        # print test progress
        print(str(i / num_obs) + " of the way done")
        # get the input and label
        input_image = Image.open(inputs[i])

        # zero out the gradients
        output = model(input_image)

        if output[0][0] > 0.5 and ground_truth[i] == 1 or output[0][0] < 0.5 and ground_truth[i] == 0:
            correct += 1
    
    print("accuracy is " + str(correct / 100))
    


            
        
test()
