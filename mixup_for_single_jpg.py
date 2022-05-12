#modified from mix.py,which is https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
#adopting the core function "forward" from the class RandomMixup from mix.py
#mixup param is 0.4

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

# Implemented as on mixup paper, page 3.
lambda_param = 0.4
res=batch_rolled.mul_(1.0 - lambda_param)
res=batch.mul_(lambda_param).add_(res)

tres=target_rolled.mul_(1.0 - lambda_param)
tres=target.mul_(lambda_param).add_(tres)

image1=np.array(batch[0]).transpose(2,1,0)
image2=np.array(batch[1]).transpose(2,1,0)
image3=np.array(batch[2]).transpose(2,1,0)

plt.imsave('mixup1.jpg',image1)
plt.imsave('mixup2.jpg',image2)
plt.imsave('mixup3.jpg',image3)
