from __future__ import print_function
import math
from tkinter.dnd import dnd_start
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Sampler
import torchvision
import torchvision.transforms as transforms
from timm.data.mixup import Mixup
from timm.utils import accuracy, ModelEma, get_state_dict
import timm.optim.optim_factory as optim_factory
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import argparse
import PIL
import os
import time
import numpy as np

from transforms import build_dataset, new_data_aug_generator
from utils import copy_files, Logger ,AverageMeter ,console_logger, save_on_master
from utils import cosine_scheduler_epoch, transform_cosine_scheduler, mask_cosine_scheduler, NativeScalerWithGradNormCount
from engine import train, test
from models.dems import DEMS_ViT

def main(args):
    print(args)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.world_size * ngpus_per_node
    if args.distributed:
        mp.spawn(main_worker,args=(ngpus_per_node, args),nprocs=args.world_size)
    else:
        main_worker(args.rank, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):    
    rank = args.rank * ngpus_per_node + gpu
    if args.distributed:
        dist.init_process_group(
            backend="nccl",
            init_method=args.init_method,
            world_size=args.world_size,
            rank=rank,
        )
        torch.distributed.barrier()
    args.rank = rank

    if args.rank == 0:
        logger_tb = Logger(args.exp_dir, '')
        logger_console = console_logger(logger_tb.log_dir, 'console')
        dst_dir = os.path.join(logger_tb.log_dir, 'code/')
        copy_files('./', dst_dir, args.exclude_file_list)
    else:
        logger_tb,logger_console = None,None

    trainset, args.n_classes = build_dataset(args, is_train='train')
    testset, _ = build_dataset(args, is_train='test')

    sampler_train = torch.utils.data.DistributedSampler(trainset) if args.distributed else None
    sampler_test = torch.utils.data.DistributedSampler(testset) if args.distributed else None

    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler_train,batch_size=bs_per_gpu, pin_memory=False,shuffle=(sampler_train is None), drop_last=True, num_workers=args.num_workers, persistent_workers = True)       
    testloader = torch.utils.data.DataLoader(testset, sampler=sampler_test,batch_size=bs_per_gpu, pin_memory=False, shuffle=False, drop_last=True, num_workers=args.num_workers, persistent_workers = True)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        if args.rank == 0:
            print('Mixup activated')
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.n_classes)

    net = DEMS_ViT(
        image_size = args.size,
        patch_size = args.patch,
        num_classes = args.n_classes,
        dim = args.dim,
        depth = args.depth,
        heads = args.num_heads,
        merge_num = args.merge_num,
        merge_layer = args.merge_layer,
        mlp_ratio = args.mlp_ratio,
        drop_path_rate = args.drop_path,
        attn_drop_rate = args.attn_drop, 
    )
    net = net.cuda(args.rank)

    args.lr = args.lr * args.bs / 256
    bs_per_gpu = int(args.bs)
    if args.distributed:
        torch.cuda.set_device(args.rank)
        bs_per_gpu = int(args.bs / args.world_size)
        args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size) 
        # net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.rank], broadcast_buffers=False)

    if args.ThreeAugment:
        trainloader.dataset.transform = new_data_aug_generator(args)
    
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    parameters = net.module.parameters() \
    if isinstance(net, nn.parallel.DistributedDataParallel) else net.parameters()
    optimizer = torch.optim.AdamW(parameters, 
                    lr=args.lr, 
                    betas=(0.9, 0.999), 
                    weight_decay=args.weight_decay) 
    
# use cosine scheduling
    scheduler = cosine_scheduler_epoch(base_value=args.lr, final_value=args.min_lr, epochs=args.n_epochs, warmup_epochs=args.warmup_epochs, stage=1)

    if args.aug_schedule:
        transform_scheduler = transform_cosine_scheduler(base=[args.aug_p[0], args.aug_p[1], args.aug_p[2]], crop_size=args.size, iter=1, epochs=args.n_epochs)
    
    if args.save_mem:
        scaler = NativeScalerWithGradNormCount()
    else:
        scaler = torch.cuda.amp.GradScaler()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.rank is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.rank)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.n_epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = scheduler[epoch]
        lr = optimizer.param_groups[0]['lr']

        if args.aug_schedule:
            trainloader.dataset.transform = transform_scheduler[epoch]
        
        trainloss = train(epoch,net,scaler,trainloader,criterion,optimizer,mixup_fn,(logger_tb, logger_console),args)  
        acc = test(epoch,net,testloader,(logger_tb, logger_console),args)
        if args.rank == 0:
            logger_tb.add_scalar('Epoch/lr', lr, epoch + 1)
            logger_tb.add_scalar('Epoch/train_Loss', trainloss, epoch + 1)
            logger_tb.add_scalar('Epoch/val_Acc', acc, epoch + 1)
        
            if logger_console is not None:  
                logger_console.info(f'Average train loss: {trainloss:.4f}')
                logger_console.info(f'Validation accuracy: {acc:.4f}')            
        
        save_dict = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'scaler': scaler.state_dict(),
        }

        if args.rank == 0:
            os.makedirs(os.path.join(logger_tb.log_dir, 'checkpoint/'), exist_ok=True)
            save_on_master(save_dict, os.path.join(logger_tb.log_dir, 'checkpoint/', 'checkpoint.pth'))
        if args.rank==0 and (epoch % args.saveckp_freq == 0) and epoch:
            save_on_master(save_dict, os.path.join(logger_tb.log_dir, 'checkpoint/', f'checkpoint{epoch:04}.pth'))

    return 

def get_args_parser():
    parser = argparse.ArgumentParser('DEMS training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)') 

    parser.add_argument('--model', default='dems_small', choices=['dems_tiny', 'dems_small'],
                        type=str, help='Name of model to train')
          
    parser.add_argument('--data_path', default='/path/to/CIFAR100', type=str, help='dataset path')
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'CALTECH101', 'FASHIONMNIST', 'EMNIST'],
                        type=str, help='dataset')

    parser.add_argument('--output_dir', default='./', help='path where to save')
    parser.add_argument('--init_method', default='tcp://localhost:17888')
    
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args_ = parser.parse_args()
    if args_.model == 'dems_tiny':
        from config.pretrain.dems_tiny_pretrain import dems_tiny_pretrain
        args = dems_tiny_pretrain()
    elif args_.model == 'dems_small':
        from config.pretrain.dems_small_pretrain import dems_small_pretrain
        args = dems_small_pretrain()

    args.bs = args_.batch_size
    args.n_epochs = args_.epochs
    args.lr = args_.lr

    args.data_root = args_.data_path
    args.dataset = args_.dataset

    args.exp_dir = args_.output_dir
    args.init_method = args_.init_method

    main(args)

    

