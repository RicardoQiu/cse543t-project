##################################################################
# This module is created to host the architecture of our network
##################################################################
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import torch.nn.functional as F
import torchvision.models as models


class styleNet(nn.Module):
    def __init__(self, args):
        self.device = torch.device("cuda" if args.cuda.is_available() else "cpu")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    def train(self, x):
        pass

    def evaluate(self, x):
        pass
