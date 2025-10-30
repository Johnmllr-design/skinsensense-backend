import torch.nn as nn
import os
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision import transforms
from model import skin_cnn
from load_data import load_dataset
from torch.optim import Adam
import torch
from dataloaders import Dataset


class trainer():

    # constructor for hyperparameter containment
    def __init__(self, epochs):
        self.num_epochs = epochs
        self.input_filepaths, self.ground_truth = load_dataset()
        self.model = skin_cnn()
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)
        self.transform = TF.Compose([TF.Resize((256, 256)), TF.ToTensor()])


    # training function, for forward/back prop loop
    def train(self):

        # training loop
        for epoch in range(0, self.num_epochs):

            # save the number of correct inferences
            correct = 0
            false_pos = 0
            false_neg = 0
            losses = 0

            # get the dataset via the dataLoader class
            dataset = Dataset(self.input_filepaths, self.ground_truth)
            trainset = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
            
            
            # enumerate over the input path and label values
            for i, items in enumerate(trainset):

                # establish the inputs as tensors
                label = items[0].float().unsqueeze(0)
                input_path = items[1][0]
                image_file = Image.open(input_path)
                input_tensor = self.transform(image_file)
                input_tensor = input_tensor.unsqueeze(0)

                # obtain the inference
                prediction = self.model(input_tensor)
                

                # zero out the gradients
                self.optimizer.zero_grad()

                # calculate the loss
                loss = self.loss_function(prediction, label)
                losses += loss.item()

                # print the progress, correctess and 
                correct_inference = prediction[0][0].item() > 0.5 and label[0][0].item() == 1 or prediction[0][0].item() < 0.5 and label[0][0].item() == 0
                if correct_inference:
                    correct += 1
                elif prediction[0][0].item() > 0.5 and label[0][0].item() < 0.5:
                    false_pos += 1
                elif prediction[0][0].item() < 0.5 and label[0][0].item() > 0.5:
                    false_neg += 1
                print(f"\rEpoch {epoch} | Image {i+1}/{len(trainset)} | Loss: {loss.item():.4f} | Correct: {correct_inference}", end="", flush=True)
                # calculate the derivatives
                loss.backward()

                # apply the calculations
                self.optimizer.step()


            # save the losses array as a text file
            with open("accuracy.txt", "w") as f:
                print("EPOCH " + str(epoch))
                f.write("accuracy at epoch " + str(epoch) + " was " + str(correct / len(trainset)))
                f.write("there were " + str(false_pos / len(trainset)) + "percent false positives")
                f.write("there were " + str(false_neg / len(trainset)) + "percent false positives")
                f.write("the average loss was " +str(losses / len(trainset)))
            
            # save the model state
            torch.save(self.model.state_dict(), f"saved_models_3/epoch_{epoch}_model.pt")

            
            

trainer = trainer(10)
trainer.train()



