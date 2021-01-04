# -*- coding: utf-8 -*-
# @Time    : 1/3/21 11:22 PM
# @Author  : Ricardo
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize


def merge_image(img_list):
    m, n, c = img_list[0].shape
    output0 = []
    for i in range(3):
        output0.append(np.concatenate((img_list[i], np.ones((20, n, c))), axis=0))

    output0 = np.concatenate(output0, axis=0)

    # output = np.concatenate((output0, np.ones((20, output0.shape[1], c)), output1), axis=0)

    return output0


img_size = 512
images = []

# content = np.float32(resize(imread('../data/content/lena_noise.jpg'), (img_size, img_size)))
# style = np.float32(resize(imread('../data/style/starry-night.jpg'), (img_size, img_size)))
res1 = np.float32(imread('../res/inter1.jpg')) / 255.
res2 = np.float32(imread('../res/inter2.jpg')) / 255.
res3 = np.float32(imread('../res/inter3.jpg')) / 255.


# images.append(content)
# images.append(style)
images.append(res1)
images.append(res2)
images.append(res3)

# print(sorted(glob('../res/*.jpg')))
#
# for file in glob('../res/*.jpg'):
#     if 'stylize' in file:
#         images.append(np.float32(imread(file)) / 255.)

imsave('../res/final2.jpg', merge_image(images))
