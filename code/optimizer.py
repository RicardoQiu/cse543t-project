##################################################################
# This module is created to host the optimizer that will be used
# to train our neural network. Currently, three optimizers are
# taken into consideration: SGD with momentum, Adam, and AdaBelief.
# we will pretty use the official implementation to make sure the
# quality.
##################################################################
import torch.optim as optim


def get_optimizer(input_img, args):
    # this line to show that input is a parameter that requires a gradient
    if args.optimizer == "L-BFGS":
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    elif args.optimizer == "Adam":
        optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer

