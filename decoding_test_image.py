#visualization for cifar100 test images.adopted from https://blog.csdn.net/winycg/article/details/106653579?msclkid=7f55e289cf5311ec879f4d7ad03632eb
import pickle as p
import numpy as np
from PIL import Image

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'fine_labels']
        X = X.reshape(50000, 3, 32, 32)
        Y = np.array(Y)
        print(Y.shape)
        return X, Y


if __name__ == "__main__":
    imgX, imgY = load_CIFAR_batch("D:/learning/neural_network/midterm_cifar100_resnet/pytorch-cifar100-master/data/cifar-100-python/train")
    with open('img_label.txt', 'a+') as f:
        for i in range(imgY.shape[0]):
            f.write('img'+str(i)+' '+str(imgY[i])+'\n')

    for i in range(imgX.shape[0]):
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB",(i0,i1,i2))
        name = "img" + str(i)+".png"
        img.save("D:/learning/neural_network/midterm_cifar100_resnet/pytorch-cifar100-master/pic_train/"+name,"png")
    print("save successfully!")
