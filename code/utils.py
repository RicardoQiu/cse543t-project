##################################################################
# This module is created to host some very useful utility functions
# used either in building the model or visualize our result
##################################################################
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def check_paths(args):
    try:
        if not os.path.exists(args.output_image):
            os.makedirs(args.output_image)
        # if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
        #     os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


# Color transfer
def transfer_color(src, dest):
    src, dest = src.clip(0, 255), dest.clip(0, 255)

    dest_gray = cv2.cvtColor(dest, cv2.COLOR_RGB2GRAY)  # 1 Extract the Destination's luminance
    src_yiq = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)  # 2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq[..., 0] = dest_gray  # 3 Combine Destination's luminance and Source's IQ/CbCr

    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2RGB)  # 4 Convert new image from YIQ back to RGB
