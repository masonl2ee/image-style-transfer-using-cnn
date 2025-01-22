# content Loss
# vgg19 feature map -> deep image representation

# style Loss
# gram matrix -> function
# MSE


import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module): # Inherit from nn.Module.
    def __init__(self, ):
        super(ContentLoss, self).__init__()

    def forward(self, x:torch.Tensor, y): # The forward function is automatically called by PyTorch, even if the user does not call it explicitly.
        # x torch.Tensor, shape (b, c, h, w) -> (b, c, h*w)
        # MSE Loss ouputs a single result, so we don't need to implement shaping proccess above. pytorch automatically aligned their(inputs) dimensions.
        loss = F.mse_loss(x, y)
        return loss

class StyleLoss(nn.Module): # Inherit from nn.Module.
    def __init__(self, ):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, x:torch.Tensor):
        """
        x: torch.Tensor, shape (b, c, h, w)
        reshape (b, c, h, w) -> (b, c, h*w)
        dim (b, N, M)
        transpose
        matrix mul
        """

        b, c, h, w = x.size()
        # reshape
        features = x.view(b, c, h*w) # (b, N, M)
        features_T = features.transpose(1, 2) # (b, M, N)
        G = torch.matmul(features, features_T)
        return G.div(b*c*h*w)

    def forward(self, x, y): # The forward function is automatically called by PyTorch, even if the user does not call it explicitly.
        # gram matrix style representation
        # MSE
        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        loss = F.mse_loss(Gx, Gy)
        return loss # Don't need to normalize, because it is already done in the gram_matrix function.