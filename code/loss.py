##################################################################
# This module is created to host all the lost functions that will
# be used.
##################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss = None

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class LaplaceFilter(nn.Module):
    @staticmethod
    def _make_kernel():
        k = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float).expand(1, 3, 3, 3)
        return k

    def __init__(self, device):
        super().__init__()
        self.k = nn.Parameter(self._make_kernel(), False).to(device)

    def forward(self, input):
        return F.conv2d(input, self.k, padding=1)


class LapStyleLoss(nn.Module):
    """
    Reimplementation of http://arxiv.org/abs/1707.01253
    """

    def __init__(self, device, content, pooling_kernel_size=2):
        super().__init__()
        self.edge_detector = nn.Sequential(
            nn.AvgPool2d(pooling_kernel_size),
            LaplaceFilter(device)
        )
        with torch.no_grad():
            self.target = self.edge_detector(content)

    def forward(self, input):
        return F.mse_loss(self.edge_detector(input), self.target)


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, input):
        x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        loss = torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))
        return loss
