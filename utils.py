# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import PIL
import sys
import time
import math
import shutil
import scipy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributed as dist
import logging
from PIL import ImageFilter, Image
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transform import ComposeWithHFlip, RandomHorizontalFlip
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg

from transforms import GaussianBlur_deit, Solarization, gray_scale, GaussianBlur

class Logger(SummaryWriter):
    def __init__(self, log_root='./', name='', logger_name=''):
        os.makedirs(log_root, exist_ok=True)
        if logger_name == '':
            date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.log_name = '{}_{}'.format(name, date)
            # self.log_name = date
            log_dir = os.path.join(log_root, self.log_name)
            super(Logger, self).__init__(log_dir, flush_secs=1)
        else:
            self.log_name = logger_name
            log_dir = os.path.join(log_root, self.log_name)
            super(Logger, self).__init__(log_dir, flush_secs=1)
        
def console_logger(log_root, logger_name) -> logging.Logger:
    log_file = logger_name + '.log'
    log_path = os.path.join(log_root, log_file)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}


    num_layers = len(model.module.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
        
def copy_files(src_dir, dst_dir, exclude_file_list):

    fnames = os.listdir(src_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for f in fnames:
        if f not in exclude_file_list:
            src = os.path.join(src_dir, f)
            if os.path.isdir(src):
                dst = os.path.join(dst_dir, f)
                print(f'copy {src} to {dst}')
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                print(f'copy {src} to {dst_dir}')
                shutil.copy(src, dst_dir)
            else:
                ValueError(f'{src} can not be copied')

    return

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)

def cosine_scheduler_epoch(base_value, final_value, epochs, warmup_epochs=0, stage = 16):
    per_epoch = np.array([])
    stage_ratio = np.array([])
    for i in np.arange(stage):
        per_epoch = np.append(per_epoch, epochs // stage)
        stage_ratio = np.append(stage_ratio, 1)
    warmup_iters = warmup_epochs
    warmup_schedule = np.empty(shape=[stage, warmup_iters])

    stage_value = np.array([])
    start_warmup_value = np.array([])

    for i in np.arange(stage):
        stage_value = np.append(stage_value, base_value * stage_ratio[i])
        start_warmup_value = np.append(start_warmup_value, final_value)

    if warmup_epochs > 0:
        for i in np.arange(stage):
            warmup_schedule[i] = np.linspace(start_warmup_value[i], stage_value[i], warmup_iters)

    schedule_all = np.array([])
    for i in np.arange(stage):
        iters = np.arange(per_epoch[i] - warmup_iters)
        schedule = np.empty(shape=[1, len(iters)])
        schedule[0] = final_value + 0.5 * (stage_value[i] - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule_all = np.append(schedule_all, warmup_schedule[i])
        schedule_all = np.append(schedule_all, schedule[0])

    assert len(schedule_all) == epochs
    return schedule_all

def mask_cosine_scheduler(base_value, final_value, epochs):

    iters = np.arange(epochs)

    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    assert len(schedule) == epochs 
    return schedule

def transform_cosine_scheduler(base, crop_size, iter, epochs):
    import torchvision.transforms as transforms
    transform_scheduler = np.array([])
    len_iter = epochs * iter
    p_color = mask_cosine_scheduler(base_value=base[0], final_value=0, epochs=len_iter)
    p_gray = mask_cosine_scheduler(base_value=base[1], final_value=0, epochs=len_iter)
    p_solar = mask_cosine_scheduler(base_value=base[2], final_value=0, epochs=len_iter)

    for index in np.arange(len_iter):
        trans_train = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1), interpolation=3),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=p_color[index]),
            transforms.RandomGrayscale(p=p_gray[index]),
            transforms.RandomSolarize(threshold=128, p=p_solar[index]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        transform_scheduler = np.append(transform_scheduler, trans_train)

    return transform_scheduler

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

from torch._six import inf
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

