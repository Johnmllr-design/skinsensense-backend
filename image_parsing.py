from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image
from torch import Tensor
import base64
import torchvision.transforms as TF
import io
from model import skin_cnn
import numpy as np
import matplotlib.pyplot as plt

def make_img():
    Grey_scale = []
    for i in range(0, 480):
        arr = [0]*480
        Grey_scale.append(arr)
    return Grey_scale
    


def to_hash(arr):
    g_hash = {}
    i = 0
    p = 0
    placed = False
    for value in arr:
        print(f"\rvalue " + str(i) + " of " + str(len(arr)), end="", flush=True) 
        if value <= 0.1:
            if 0.1 not in g_hash.keys():
                placed = True
                g_hash[0.1] = [value]
            else:
                placed = True
                g_hash[0.1].append(value)
        elif value > 0.1 and value <= 0.2:
            if 0.2 not in g_hash.keys():
                placed = True
                g_hash[0.2] = [value]
            else:
                g_hash[0.2].append(value)
                placed = True
        elif value > 0.2 and value <= 0.3:
            if 0.3 not in g_hash.keys():
                placed = True
                g_hash[0.3] = [value]
            else:
                g_hash[0.3].append(value)
                placed = True
        elif value > 0.3 and value <= 0.4:
            if 0.4 not in g_hash.keys():
                g_hash[0.4] = [value]
                placed = True
            else:
                g_hash[0.4].append(value)
                placed = True
        elif value > 0.4 and value <= 0.5:
            if 0.5 not in g_hash.keys():
                placed = True
                g_hash[0.5] = [value]
            else:
                g_hash[0.5].append(value)
                placed = True
        elif value > 0.5 and value <= 0.6:
            if 0.6 not in g_hash.keys():
                placed = True
                g_hash[0.6] = [value]
            else:
                placed = True
                g_hash[0.6].append(value)
        elif value > 0.6 and value <= 0.7:
            if 0.7 not in g_hash.keys():
                placed = True
                g_hash[0.7] = [value]
            else:
                g_hash[0.7].append(value)
                placed = True
        elif value > 0.7 and value <= 0.8:
            if 0.8 not in g_hash.keys():
                placed = True
                g_hash[0.8] = [value]
            else:
                placed = True
                g_hash[0.8].append(value)
        elif value > 0.8 and value <= 0.9:
            if 0.9 not in g_hash.keys():
                placed = True
                g_hash[0.9] = [value]
            else:
                placed = True
                g_hash[0.9].append(value)
        elif value > 0.9 and value <= 1:
            if 1 not in g_hash.keys():
                placed = True
                g_hash[1] = [value]
            else:
                placed = True
                g_hash[1].append(value)
        i += 1
        if placed:
            p += 1
    print("placed " + str(p) + " items")
    return g_hash


get_input = TF.Compose([TF.Resize((480, 480)), TF.ToTensor()])



image_file = Image.open("ISIC_0000142_downsampled_640x480.jpg")
tensor = get_input(image_file)

red = []
green = []
blue = []

for i in range(0, 480):
    for j in range(0, 480):
        red.append(tensor[0][i][j])
        green.append(tensor[1][i][j])
        blue.append(tensor[2][i][j])

print(len(red))
print(len(green))
print(len(blue))



r_hash = to_hash(red)
# g_hash = to_hash(green)
# b_hash = to_hash(blue)

ind = 0.1
maxi = 0
max_bucket = -1
for key in r_hash.keys():
    if len(r_hash[key]) > maxi:
        maxi = len(r_hash[key])
        max_bucket = key



img = make_img()
for i in range(0, 480):
    for j in range(0, 480):
        if tensor[0][i][j] > (max_bucket - 0.1) and tensor[0][i][j] <= max_bucket:
            img[i][j] = 0
        else:
            img[i][j] = 1

np_arr = np.array(img)
plt.imshow(np_arr, cmap='gray')
plt.axis('off')
plt.show()







# data = np.load("your_array.npy")
# Example demo data:
def display_data(data):
    # make a histogram
    counts, edges = np.histogram(data, bins=100)
    centers = (edges[:-1] + edges[1:]) / 2

    # find where values are most concentrated (top 20% of counts)
    threshold = np.percentile(counts, 80)
    high_density = counts >= threshold

    # plot
    plt.figure(figsize=(10,5))
    plt.bar(centers, counts, width=edges[1]-edges[0], color='gray', alpha=0.6)
    plt.bar(centers[high_density], counts[high_density], width=edges[1]-edges[0], color='red', alpha=0.7)
    plt.title("Concentration of Values (High-density bins in red)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

