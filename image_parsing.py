from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import to_hash
from PIL import Image
from torch import Tensor
import base64
import torchvision.transforms as TF
import io
import os
from model import skin_cnn
import numpy as np
import matplotlib.pyplot as plt


    
def get_channel(tensor: Tensor, channel):
        ret = []
        for i in range(0, len(tensor[0])):
            for j in range(0, len(tensor[0][0])):
                ret.append(tensor[channel][i][j])
        return ret


def get_image_sizes():
    input_path = "/Users/johnmiller/Desktop/skin_dataset_resized/test_set/benign/"
    benign_filenames = os.listdir("/Users/johnmiller/Desktop/skin_dataset_resized/test_set/benign")
    abnormality_sizes = []
    for i in range(0, 100):
        print("on image " + str(i))
        get_input = TF.Compose([TF.Resize((480, 480)), TF.ToTensor()])
        to_image = TF.ToPILImage()
        image_file = Image.open(input_path + benign_filenames[i])
        tensor = get_input(image_file)


        red_channel = get_channel(tensor, 0)
        hash = to_hash(red_channel)
        maxItem1 = 0
        maxVal1 = 0
        for key in hash.keys():
            if hash[key] > maxVal1:
                maxItem1 = key
                maxVal1 = hash[key]


        blue_channel = get_channel(tensor, 1)
        hash = to_hash(blue_channel)
        maxItem2 = 0
        maxVal2 = 0
        for key in hash.keys():
            if hash[key] > maxVal2:
                maxItem2 = key
                maxVal2 = hash[key]

        green_channel = get_channel(tensor, 2)
        hash = to_hash(green_channel)
        maxItem3 = 0
        maxVal3 = 0
        for key in hash.keys():
            if hash[key] > maxVal3:
                maxItem3 = key
                maxVal3 = hash[key]

        abnormality_size = 0
        for i in range(0, 480):
            for j in range(0, 480):
                if tensor[0][i][j] < maxItem1 and tensor[0][i][j] > maxItem1 - 0.1 or tensor[0][i][j] < maxItem2 and tensor[0][i][j] > maxItem2 - 0.1 or tensor[0][i][j] < maxItem3 and tensor[0][i][j] > maxItem3 - 0.1 :
                    tensor[0][i][j] = 0
                    tensor[1][i][j] = 0
                    tensor[2][i][j] = 0
                else:
                    abnormality_size += 1

        abnormality_sizes.append(abnormality_size)
    return sum(abnormality_sizes) /  len(abnormality_sizes)

print("the average abnormality size is ", get_image_sizes())





