import time
import torch
from timm.utils import accuracy
from utils import AverageMeter

def train(epoch,net,scaler,trainloader,criterion,optimizer,mixup_fn,loggers,args):
    net.train()

    logger_tb, logger_console = loggers

    data_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    num_iter = len(trainloader)
    niter_global = epoch * num_iter
    end = time.time()

    for batch_idx, (inputsall, targets) in enumerate(trainloader):
        inputs, targets = inputsall.cuda(non_blocking = True), targets.cuda(non_blocking = True)
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        data_time.update(time.time() - end)      
        with torch.cuda.amp.autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        if args.save_mem:
            scaler(loss, optimizer, parameters=net.parameters(),
                    update_grad=(epoch * num_iter + batch_idx + 1) % 2 == 0)
            if (epoch * num_iter + batch_idx + 1) % 2 == 0:
                optimizer.zero_grad()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_size = inputs.shape[0]
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)

        niter_global += 1
        if args.rank == 0:
            logger_tb.add_scalar('Iter/loss', losses.val, niter_global)

        if logger_console is not None and args.rank == 0 and (batch_idx % 20 == 0):  
            lr = optimizer.param_groups[0]['lr']
            logger_console.info(f'Epoch [{epoch}][{batch_idx+1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss: {losses.val:.3f}({losses.avg:.3f})')

    return losses.avg

def test(epoch,net,testloader,loggers,args):
    net.eval()
    logger_tb, logger_console = loggers
    accs = AverageMeter('Acc', ':.4e')

    num_iter = len(testloader)
    niter_global = epoch * num_iter

    with torch.no_grad():
        for batch_idx, (inputsall, targets) in enumerate(testloader):
            inputs, targets = inputsall.cuda(non_blocking = True), targets.cuda(non_blocking = True)
            outputs = net(inputs)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = inputs.shape[0]
            accs.update(acc1.item(), batch_size)

            batch_size = inputs.shape[0]
            accs.update(acc1,batch_size)

            niter_global += 1
            if args.rank == 0:
                logger_tb.add_scalar('Iter/val_acc', accs.val, niter_global)

            if logger_console is not None and args.rank == 0 and (batch_idx % 10 == 0):  
                logger_console.info(f'Epoch [{epoch}][{batch_idx+1}/{num_iter}] - '
                            f'acc: {accs.val:.3f}({accs.avg:.3f})')
    return accs.avg