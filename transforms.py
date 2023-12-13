import random
import numpy as np
import PIL
from PIL import ImageFilter, ImageOps

import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import RandomResizedCropAndInterpolation

def build_dataset(args, is_train):
    transform = build_transform(args, is_train)

    if args.dataset == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root=args.data_root, train=False if is_train == 'test' else True, download=False, transform=transform)
        nb_classes = 10
    elif args.dataset == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root=args.data_root, train=False if is_train == 'test' else True, download=False, transform=transform)
        nb_classes = 100
    elif args.dataset == 'FASHIONMNIST':
        dataset = torchvision.datasets.FashionMNIST(root=args.data_root, train=False if is_train == 'test' else True, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'EMNIST':
        dataset = torchvision.datasets.EMNIST(root=args.data_root, split = 'mnist', train=False if is_train == 'test' else True, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'CALTECH101':
        root = os.path.join(args.data_root, 'test') if is_train == 'test' else os.path.join(args.data_root, 'train')
        
        dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
        nb_classes = 101

def build_transform(args, is_train):
    if 'MNIST' in args.dataset:
        mean, std = [0.5], [0.5]
    else:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    if is_train == 'train':
        primary_tfl = [
            transforms.RandomResizedCrop(args.size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip()
        ]

        secondary_tfl = [
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomSolarize(threshold=128, p=0.2)
        ]
        if 'MNIST' not in args.dataset:
            secondary_tfl.append(transforms.RandomGrayscale(p=0.2))

        final_tfl = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        transform = transforms.Compose(primary_tfl+secondary_tfl+final_tfl)
    elif is_train == 'finetune':
        transform = transforms.Compose([
                transforms.Resize(args.size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    elif is_train == 'test':
        if args.dataset == 'Caltech101':
            transform = transforms.Compose([
                transforms.Resize((args.resize, args.resize), PIL.Image.BILINEAR),
                transforms.CenterCrop((args.size, args.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(args.size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    
    return transform

def get_patch_contrast_labels(img_size, patch_size1, patch_size2, \
    stride2, is_hflip=False):
    # 这里假设view1, 是小尺度patch, 其patch embedding进行卷积的stride1 = patch_size1
    # view2 为大尺度patch, 其patch_size2为view1的patch size1的整数倍
    # 其patch embedding进行卷积的stride2为view1的patch size1的整数倍
    stride1 = patch_size1
    grid_size1 = img_size // stride1
    grid_size2 = (img_size - patch_size2) // stride2 + 1
    
    i = torch.div(torch.arange(0, img_size - patch_size2 + stride2, \
        step=stride2, dtype=torch.long), patch_size1, rounding_mode='trunc')
    j = i.clone() # shape: (grid_size2, )
    
    n = patch_size2 // patch_size1 # view2中一个patch能够覆盖view1中的个数：n^2
    r = i.view(grid_size2, 1) + torch.arange(0, n).view(1, n) # shape: (grid_size2, n)
    c = j.view(grid_size2, 1) + torch.arange(0, n).view(1, n) # shape: (grid_size2, n)
    #r = r.repeat(batch, 1, 1)
    #c = c.repeat(batch, 1, 1)

    #for index in np.arange(len(is_hflip)):
    if is_hflip:
            c = grid_size1 - 1 - c
    r = r.view(grid_size2, 1, n, 1)
    c = c.view(1, grid_size2, 1, n)
    t = r * grid_size1 + c # (grid_size2, grid_size2, n, n)
    t = t.view(grid_size2 * grid_size2, n * n)
    return t

# inherit from DeiT
def new_data_aug_generator(args = None):
    img_size = args.size
    remove_random_resized_crop = False
    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    # mean, std = [0.5], [0.5]
    primary_tfl = []
    scale=(0.2, 1.0)
    interpolation='bicubic'
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(args.resize, interpolation=3),
            transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

        
    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
                                              Solarization(p=1.0),
                                              GaussianBlur_deit(p=1.0)
                                              ])]

    if args.color_jitter is not None and not args.color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(0.3, 0.3, 0.3))
    final_tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)