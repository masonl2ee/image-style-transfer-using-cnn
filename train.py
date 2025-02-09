# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image

from models import StyleTransfer
from loss import ContentLoss, StyleLoss


mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

def pre_processing(image: Image.Image) -> torch.Tensor:
    # Resize image (512 * 512)
    # Image -> Tensor
    # Normalize
    preprocessing = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean, std) # Lambda x : (x - mean) / std
    ])
    # (c, h, w)
    # (1, c, h, w)
    image_tensor:torch.Tensor = preprocessing(image)

    return image_tensor.unsqueeze(0)

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

    style_image = Image.open('img/style.jpg')
    style_image = pre_processing(style_image)
    ## pre processing

    # Load model
    style_transfer = StyleTransfer().eval()

    # Load Loss
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    #hyper parameter
    alpha = 1
    beta = 1
    lr = 0.01

    # device setting

    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
    style_transfer = style_transfer.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    x = torch.randn(1,3,512,512).to(device)
    x.requires_grad_(True)

    # setting optimizer
    optimizer = optim.Adam([x], lr=lr)


    # train Loop
    steps = 1000
    for step in range(steps):

        ## content representation (x, content_image)
        ## style representation (x, style_image)
        x_content_list = style_transfer(x, 'content')
        y_content_list = style_transfer(content_image, 'content')

        x_style_list = style_transfer(x, 'style')
        y_style_list = style_transfer(style_image, 'style')

        ## Loss_content, Loss_style
        loss_c = 0
        loss_s = 0
        loss_total = 0

        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c += content_loss(x_content, y_content)
        loss_c = alpha*loss_c

        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_s += style_loss(x_style, y_style)
        loss_s = beta*loss_s

        loss_total = loss_c + loss_s
        
        ## optimizer step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        ## Loss print
        print(loss_c)
        print(loss_s)
        print(loss_total)
        
        ## post processing
        ## image gen output save

if __name__ == "__main__":
    train_main()