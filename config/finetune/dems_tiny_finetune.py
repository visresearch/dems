import os
import argparse

def dems_tiny_finetune():
    args = argparse.Namespace()

    args.depth = 12
    args.num_heads = 3
    args.dim = 192
    args.mlp_ratio = [4., 4., 4.]

    args.resize = 300 # set for caltech101
    args.size = 32  # input size; 256 for large resolution images
    args.patch = 2  # patch size; 16 for 256 input size

    args.warmup_epochs = 0
    args.min_lr = 1e-6
    args.weight_decay = 0.05
    args.drop_path = 0.2

    args.distributed = True
    args.rank = 0
    args.world_size = 1
    args.num_workers= 8
    args.init_method = 'tcp://localhost:17997'

    args.resume = ''
    args.start_epoch = 0

    args.use_mixup = True
    args.smoothing = 0.1
    args.mixup = 0.8
    args.cutmix = 1.0
    args.cutmix_minmax = None # float
    args.mixup_prob = 1.0
    args.mixup_switch_prob = 0.5
    args.mixup_mode = 'batch'

    args.exclude_file_list = ['__pycache__', '.gitignore']

    return args