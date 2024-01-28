import os
import argparse

def dems_small_pretrain():
    args = argparse.Namespace()

    args.depth = 12
    args.num_heads = 6
    args.dim = 384
    args.mlp_ratio = [4., 4., 4.]

    args.resize = 300 # set for caltech101
    args.size = 32  # input size; 256 for large resolution images
    args.patch = 2  # patch size; 16 for 256 input size

    args.merge_num = [64, 32]
    args.merge_layer = [6, 10]

    args.ThreeAugment = False
    args.aug_schedule = True
    args.aug_p = [0.8, 0.8, 0.8]
    
    args.warmup_epochs = 10
    args.min_lr = 1e-6
    args.weight_decay = 0.05
    args.drop_path = 0.15
    args.attn_drop = 0.05

    args.distributed = True
    args.rank = 0
    args.world_size = 1
    args.num_workers= 8
    args.init_method = 'tcp://localhost:17997'

    args.save_mem = False
    args.resume = ''
    args.start_epoch = 0
    args.saveckp_freq = 50

    args.use_mixup = True
    args.smoothing = 0.1
    args.mixup = 0.8
    args.cutmix = 1.0
    args.cutmix_minmax = None # float
    args.mixup_prob = 1.0
    args.mixup_switch_prob = 0.5
    args.mixup_mode = 'batch'

    args.exclude_file_list = ['__pycache__', '.gitignore', 'out']

    return args