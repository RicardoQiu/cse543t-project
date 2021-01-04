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
    optimizer = None
    if args.optimizer == "L-BFGS":
        optimizer = optim.LBFGS([input_img.requires_grad_()], lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam([input_img.requires_grad_()], lr=args.lr, betas=(0.99, 0.999), eps=1e-6)
    elif args.optimizer == "Heavy-ball":
        optimizer = optim.SGD([input_img.requires_grad_()], momentum=0.9, lr=args.lr, nesterov=True)
    else:
        print('no such optimizer')

    return optimizer
