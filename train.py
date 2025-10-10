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

def train():

    # hyperparameters
    epochs = 100
    model = torch.nn.LSTMCell(input_size=1, hidden_size=1)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    # tensors for saving the losses
    losses = []
    all_losses = []

   # training loop
    for epoch in range(0, epochs):
        correct = 0

        # get the dataset. load_dataset shuffles data automatically such that no
        # two inputs arrays are identical
        inputs, ground_truth = load_dataset()
        epoch_loss = []
        
        for i in range(0, len(inputs)):
            # get the input and label
            input_image = Image.open(inputs[i])
            label_tensor = Tensor([[ground_truth[i]]])

            # zero out the gradients
            optimizer.zero_grad()

            # forward pass
            output = model(input_image)

            # check model output
            if output[0][0] > 0.5 and ground_truth[i] == 1 or output[0][0] < 0.5 and ground_truth[i] == 0:
                correct += 1

            # loss calculation and saving
            loss = loss_function(output, label_tensor)
            epoch_loss.append(loss.item())
            all_losses.append(loss.item())

            # backpropagate
            loss.backward()
            optimizer.step()
            print(f"\rEpoch {epoch+1}/{epochs} - Progress: {i/len(inputs):.2%} - Loss: {loss.item():.4f}", end="", flush=True) 

        losses.append(sum(epoch_loss) / len(epoch_loss))           
        # save the models
        print()
        print("accuracy of epoch " + str(epoch) +" was " + str(correct / len(inputs)))
        print()
        model_path = os.path.join("saved_models_2", f"skin_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
    
    # save the losses array as a text file
    with open("losses.txt", "w") as f:
        for i, loss in enumerate(losses):
            f.write(f"epoch {i}: {loss}\n")
    
    # save the all losses array as a tensor
    torch.save(torch.tensor(all_losses), "all_losses.pt")



            
            




train()
