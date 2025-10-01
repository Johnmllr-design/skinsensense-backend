from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision
from model import skin_cnn

model = skin_cnn()
print("here")
toTensor = torchvision.transforms.ToTensor()
fpath = "/Users/johnmiller/Desktop/skin_dataset_resized/train_set/"
benign_file_names = os.listdir("/Users/johnmiller/Desktop/skin_dataset_resized/train_set/benign")
malignant_filenames = os.listdir("/Users/johnmiller/Desktop/skin_dataset_resized/train_set/malignant")
print(malignant_filenames[0])

image_file = Image.open(os.path.join(fpath, "malignant", malignant_filenames[0]))
tensor = toTensor(image_file)
tensor = tensor.unsqueeze(0)  # Add batch dimension: [C, H, W] -> [1, C, H, W]
model(tensor)

