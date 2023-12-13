from __future__ import print_function
import math
from tkinter.dnd import dnd_start
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from timm.data.mixup import Mixup
from timm.utils import accuracy
import timm.optim.optim_factory as optim_factory
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from transforms import build_dataset
from utils import copy_files, ExponentialMovingAverage, Logger ,AverageMeter ,console_logger, save_on_master, param_groups_lrd
from utils import cosine_scheduler_epoch, transform_cosine_scheduler, mask_cosine_scheduler, GaussianBlur
from engine import train, test
from models.dems import DEMS_ViT
# from models.std_vit import ViT

def main(args):
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.world_size * ngpus_per_node
    if args.distributed:
        mp.spawn(main_worker,args=(ngpus_per_node, args),nprocs=args.world_size)
    else:
        main_worker(args.rank, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
# take in args     
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

    trainset, args.n_classes = build_dataset(args, is_train='finetune')
    testset, _ = build_dataset(args, is_train='test')

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
        mlp_ratio = args.mlp_ratio,
        merge_num = args.merge_num,
        merge_layer = args.merge_layer,
        drop_path_rate = args.drop_path,
    )
    net = net.cuda(args.rank)

    args.lr = args.lr * args.bs / 256
    bs_per_gpu = int(args.bs)
    if args.distributed:
        torch.cuda.set_device(args.rank)
        bs_per_gpu = int(args.bs / args.world_size)
        args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size) 
        #net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.rank])

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
    
    lr_scheduler = cosine_scheduler_epoch(base_value=args.lr, final_value=args.min_lr, epochs=args.n_epochs, warmup_epochs=args.warmup_epochs, stage=1)

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
            # old_dict = checkpoint['model']
            # state_dict_new = dict()
            # for key, val in old_dict.items():
            #     state_dict_new[key[7:]] = val
            # net.load_state_dict(state_dict_new)
            
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.rank == 0:
            print(f'Loading parameters from: {args.pretrained_weight}')

        state_dict_old = torch.load(args.pretrained_weight, map_location=f"cuda:{args.rank}")
        state_dict_new = dict()
        #print(state_dict_old['model_ema'].keys())
        for key, val in state_dict_old['model'].items():
            #state_dict_new[key] = val
            state_dict_new[key] = val
        
        missing_keys, unexpected_keys = net.load_state_dict(state_dict_new, strict = False)
        if args.rank == 0:
            print('missing_keys:', missing_keys)
            print('unexpected_keys:', unexpected_keys)

##### Training
    best_acc = 0.
    best_epoch = 0
    for epoch in range(args.start_epoch, args.n_epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[epoch]

        lr = optimizer.param_groups[0]['lr']

        trainloss = train(epoch,net,scaler,trainloader,criterion,optimizer,mixup_fn,(logger_tb, logger_console),args)    
        acc = test(epoch,net,testloader,(logger_tb, logger_console),args)
        
        if args.rank == 0:
            logger_tb.add_scalar('Epoch/lr', lr, epoch + 1)
            logger_tb.add_scalar('Epoch/train_Loss', trainloss, epoch + 1)
            logger_tb.add_scalar('Epoch/val_Acc', acc, epoch + 1)

            if logger_console is not None:  
                logger_console.info(f'Average train loss: {trainloss:.4f}')
                logger_console.info(f'Validation accuracy: {acc:.4f}') 

        if args.rank==0:
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

                save_dict = {
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'scaler': scaler.state_dict(),
                }
                    
                os.makedirs(os.path.join(logger_tb.log_dir, 'checkpoint/'), exist_ok=True)
                save_on_master(save_dict, os.path.join(logger_tb.log_dir, 'checkpoint/', 'best_checkpoint.pth'))
            if logger_console is not None:  
                logger_console.info(f'best acc: {best_acc:.3f} at Epoch {best_epoch}')

    return 

def get_args_parser():
    parser = argparse.ArgumentParser('DEMS training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)') 

    parser.add_argument('--model', default='dems_small', choices=['dems_tiny', 'dems_small'],
                        type=str, help='Name of model to train')
          
    parser.add_argument('--data_path', default='', type=str, help='dataset path')
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'CALTECH101', 'FASHIONMNIST', 'EMNIST'],
                        type=str, help='dataset')

    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--init_method', default='tcp://localhost:17888')

    parser.add_argument('--pretrained_weight', default='', help='path where to load pretrained weight')
    
    return parser

if __name__ == '__main__':
    args_ = parser.parse_args()
    if args_.model == 'dems_tiny':
        from config.finetune.dems_tiny_finetune import dems_tiny_finetune
        args = dems_tiny_finetune()
    elif args_.model == 'dems_small':
        from config.finetune.dems_small_finetune import dems_small_finetune
        args = dems_small_finetune()

    args.bs = args_.batch_size
    args.n_epochs = args_.epochs
    args.lr = args_.lr

    args.data_root = args_.data_path
    args.dataset = args_.dataset

    args.exp_dir = args_.output_dir
    args.init_method = args_.init_method

    args.pretrained_weight = args_.pretrained_weight
    
    main(args)