import torch.nn as nn
import torchvision
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision import transforms
from model import skin_cnn
from load_data import load_dataset
from torch.optim import Adam

def train():
    epochs = 100
    model = skin_cnn()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    inputs, ground_truth = load_dataset()
    for epoch in range(0, epochs):
        correct = 0
        for i in range(0, 10):
            # get the input and label
            input_image = Image.open(inputs[i])
            label_tensor = Tensor([[ground_truth[i]]])

            # zero out the gradients
            optimizer.zero_grad()

            # forward pass
            output = model(input_image)
            print("the output is " + str(output[0][0]) +" and the label was " + str(ground_truth[i]))
            if output[0][0] > 0.5 and ground_truth[i] == 1 or output[0][0] < 0.5 and ground_truth[i] == 0:
                correct += 1

            # loss calculation
            loss = loss_function(output, label_tensor)

            # backpropagate
            loss.backward()
            optimizer.step()
        print("accuracy of epoch " + str(epoch) +" was " + str(correct / 10))
    


            
            




train()
