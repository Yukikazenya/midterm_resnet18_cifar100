# midterm_resnet18_cifar100

This is my midterm homework.The main code is from https://github.com/weiaicunzai/pytorch-cifar100. I will only upload the modified python file to this project. All the unchanged files are not uploaded.
train.py and util.py : add train dataloader for cutmix,cutout and mixup
predict.py: show three pictures in test set and get their prediction from the model.
mix.py:modified from torchvision,which is https://github.com/pytorch/vision/blob/main/references/classification/transforms.py, including method for cutmix,cutout and mixup
cutout_for_single_jpg.py:show the cutout for three selected pictures
mixup_for_single_jpg.py:show the mixup for three selected pictures
cutmix_for_single_jpg.py:show the cutmix for three selected pictures
