import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np

image1=plt.imread('D:/learning/neural_network/midterm_cifar100_resnet/pytorch-cifar100-master/pic_train/img0.png')
image2=plt.imread('D:/learning/neural_network/midterm_cifar100_resnet/pytorch-cifar100-master/pic_train/img1.png')
image3=plt.imread('D:/learning/neural_network/midterm_cifar100_resnet/pytorch-cifar100-master/pic_train/img2.png')

target=[19,29,0]

image1=image1.transpose(2,0,1)
image2=image2.transpose(2,0,1)
image3=image3.transpose(2,0,1)
batch=np.stack((image1,image2,image3))
batch=torch.tensor(batch)
target=torch.tensor(target)
target = torch.nn.functional.one_hot(target, num_classes=100).to(dtype=batch.dtype)

batch_rolled = batch.roll(1, 0)
target_rolled = target.roll(1, 0)

lambda_param = 0.4

#_, H, W = F.get_dimensions(batch)
H=32
W=32

r_x = torch.randint(W, (1,))
r_y = torch.randint(H, (1,))

r = 0.5 * math.sqrt(1.0 - lambda_param)
r_w_half = int(r * W)
r_h_half = int(r * H)

x1 = int(torch.clamp(r_x - r_w_half, min=0))
y1 = int(torch.clamp(r_y - r_h_half, min=0))
x2 = int(torch.clamp(r_x + r_w_half, max=W))
y2 = int(torch.clamp(r_y + r_h_half, max=H))

batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

target_rolled.mul_(1.0 - lambda_param)
target.mul_(lambda_param).add_(target_rolled)

image1=np.array(batch[0]).transpose(2,1,0)
image2=np.array(batch[1]).transpose(2,1,0)
image3=np.array(batch[2]).transpose(2,1,0)

plt.imsave('cutmix1.jpg',image1)
plt.imsave('cutmix2.jpg',image2)
plt.imsave('cutmix3.jpg',image3)

