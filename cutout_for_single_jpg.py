import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np
import random

image1=plt.imread('D:/learning/neural_network/midterm_cifar100_resnet/pytorch-cifar100-master/pic_train/img0.png')
image2=plt.imread('D:/learning/neural_network/midterm_cifar100_resnet/pytorch-cifar100-master/pic_train/img1.png')
image3=plt.imread('D:/learning/neural_network/midterm_cifar100_resnet/pytorch-cifar100-master/pic_train/img2.png')

target=[19,29,0]

center_i=random.randint(0,32)
center_j=random.randint(0,32)
y1 = np.clip(center_i - 8 // 2, 0, 32).item()
y2 = np.clip(center_i + 8 // 2, 0, 32).item()
x1 = np.clip(center_j - 8 // 2, 0, 32).item()
x2 = np.clip(center_j + 8 // 2, 0, 32).item()

image1[x1:x2,y1:y2,:]=0
image2[x1:x2,y1:y2,:]=0

plt.imsave('cutout1.jpg',image1)
plt.imsave('cutout2.jpg',image2)
plt.imsave('cutout3.jpg',image3)

