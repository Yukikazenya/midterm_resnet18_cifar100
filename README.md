# midterm&final: Resnet18 and vit for cifar100

Updating for final project:add code for vit in util.py

This is my midterm and final homework.The main code is from https://github.com/weiaicunzai/pytorch-cifar100. I will only upload the modified python file to this project. All the unchanged files are not uploaded.

train.py and util.py : add train dataloader for cutmix,cutout and mixup

predict.py: show three pictures in test set and get their prediction from the model.

mix.py:modified from torchvision,which is https://github.com/pytorch/vision/blob/main/references/classification/transforms.py, including method for cutmix,cutout and mixup

cutout_for_single_jpg.py:show the cutout for three selected pictures

mixup_for_single_jpg.py:show the mixup for three selected pictures

cutmix_for_single_jpg.py:show the cutmix for three selected pictures

decoding_test_image.py:from https://blog.csdn.net/winycg/article/details/106653579?msclkid=7f55e289cf5311ec879f4d7ad03632eb ,used to decode the test image.


