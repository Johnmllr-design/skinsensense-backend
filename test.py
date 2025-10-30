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
from load_data import load_testset
from torch.optim import Adam
import torch

def test(num_obs: int):
    # hyperparameters
    model = skin_cnn()
    model.load_state_dict(torch.load("saved_models_2/skin_model_epoch_9.pth"))
    print("getting the inoputs and ground truth")
    inputs, ground_truth = load_testset()
    print(len(inputs))
    inputs = inputs[0:num_obs]
    ground_truth = ground_truth[0:num_obs]
    correct = 0
    false_negatives = 0
    false_positives = 0

    for i in range(0, num_obs):
        # print test progress
        print(f"\rProgress: " + str(i/ num_obs ), end="", flush=True)            
        # get the input and label
        input_image = Image.open(inputs[i])

        # zero out the gradients
        output = model(input_image)
        print("prediction is " + str(torch.sigmoid(output[0][0])))

        if output[0][0] > 0.5 and ground_truth[i] == 1 or output[0][0] < 0.5 and ground_truth[i] == 0:
            correct += 1
        if output[0][0] < 0.5 and ground_truth[i] == 1:
            false_negatives += 1
        if output[0][0] > 0.5 and ground_truth[i] == 0:
            false_positives += 1

    print()
    print("accuracy was " + str(correct / num_obs))
    print()
    print("there were  " + str(false_negatives / num_obs) + " percent false negatives")
    print("there were " + str(false_positives / num_obs) + " percent false positives")
    print()
    return correct / num_obs



# model verification and validation via a test loop 
# that takes an average over 10 trials of the test set

test(100)            
        
# if __name__ == "__main__":
#     model_accuracies = []
#     for i in range(0, 10):
#         model_accuracies.append(test(100))
#     print("the average accuracy across 10 test epochs is " + str(sum(model_accuracies) / len(model_accuracies)))