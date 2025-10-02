import random
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision
from model import skin_cnn



def load_dataset():
    flip = random.random()
    benign_filenames = os.listdir("/Users/johnmiller/Desktop/skin_dataset_resized/train_set/benign")
    malignant_filenames = os.listdir("/Users/johnmiller/Desktop/skin_dataset_resized/train_set/malignant")
    
    # if flip is greater than 0.5, flip the dataset to avoid overfitting the last values
    if flip > 0.5:
        benign_filenames = benign_filenames[::-1]
        malignant_filenames = malignant_filenames[::-1]

    bPath = "/Users/johnmiller/Desktop/skin_dataset_resized/train_set/benign/"
    mPath = "/Users/johnmiller/Desktop/skin_dataset_resized/train_set/malignant/"
    bInd = 0
    mInd = 0
    data = []
    ground_truth = []
    while bInd < len(benign_filenames) and mInd < len(malignant_filenames):
        rand = random.random()
        if rand < 0.56:
            data.append(bPath + benign_filenames[bInd])
            ground_truth.append(0)
            bInd += 1
        else:
            data.append(mPath + malignant_filenames[mInd])
            ground_truth.append(1)
            mInd += 1
        
    return [data, ground_truth]


