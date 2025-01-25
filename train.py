# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image


mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

def pre_processing(image: Image.Image) -> torch.Tensor:
    # Resize image (512 * 512)
    # Image -> Tensor
    # Normalize
    preprocessing = T.Compose([
        T.Resize(512, 512),
        T.ToTensor(),
        T.Normalize(mean, std) # Lambda x : (x - mean) / std
    ])

    return preprocessing(image)

def post_processing(tensor:torch.Tensor) -> Image.Image:

    # shape : b, c, h, w
    image = tensor.to('cpu').detach().numpy()
    # shape : c, h, w
    image = image.squeeze()
    # shape : h, w, c
    image = image.transpose(1, 2, 0)
    # de norm
    image = image*std + mean
    # clip
    image = image.clip(0, 1)*255
    # dtype uint8
    image = image.astype(np.uint8)
    # numpy -> Image
    return Image.fromarray(image)
    

def train_main():
    # Load data
    content_image = Image.open('img/content.jpg')
    content_image = pre_processing(content_image)
    ## pre processing
    ## post processing

    # Load model

    # Load Loss

    # setting optimizer

    # train Loop
    ## Loss print
    ## image gen output save
    pass

if __name__ == "__main__":
    train_main()