import torch
from PIL import Image
import torchvision.transforms as TF
from torch import Tensor


# A class to 
class Dataset(torch.utils.data.Dataset):

    # initialization
    def __init__(self, ground_truth, inputs):
        self.ground_truth = ground_truth
        self.inputs = inputs

    # number of samples
    def __len__(self):
        return len(self.inputs)

    # return the tensors representing the current index
    def __getitem__(self, index):

        # get the path to the image and also the ground truth value
        image_path = self.inputs[index]
        label_value = self.ground_truth[index]

        
        return image_path, label_value
