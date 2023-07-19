import enum
import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
from torchvision import datasets, transforms
from torch.profiler import profile, record_function, ProfilerActivity

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
import timm

import utils

"""
'deit3_base_patch16_224', 'deit3_base_patch16_224_in21ft1k', 'deit3_base_patch16_384', 'deit3_base_patch16_384_in21ft1k', 'deit3_huge_patch14_224', 'deit3_huge_patch14_224_in21ft1k', 'deit3_large_patch16_224', 'deit3_large_patch16_224_in21ft1k', 'deit3_large_patch16_384', 'deit3_large_patch16_384_in21ft1k', 'deit3_medium_patch16_224', 'deit3_medium_patch16_224_in21ft1k', 'deit3_small_patch16_224', 'deit3_small_patch16_224_in21ft1k', 'deit3_small_patch16_384', 'deit3_small_patch16_384_in21ft1k', 
'deit_base_distilled_patch16_224', 'deit_base_distilled_patch16_384', 'deit_base_patch16_224', 'deit_base_patch16_384', 'deit_small_distilled_patch16_224', 'deit_small_patch16_224', 'deit_tiny_distilled_patch16_224', 'deit_tiny_patch16_224', 
"""

train_config = {
    'data_loader': '',
    'model': '',
    'device': '',
    'epoch': '',
    'optimizer': '',
    'checkpoint_path': '',
    'log_path': './log/DeiT/DeiT-S-patch16_224'
}



torch.set_num_threads(32)

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    # parser.add_argument('--cfg', type=str, required=True,
    #                     metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, help='path to dataset',
                        default="/data")
    parser.add_argument('--pretrained', required=False,
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    parser.add_argument('--use-sync-bn', action='store_true',
                        default=False, help='sync bn')
    parser.add_argument('--use-wandb', action='store_true',
                        default=False, help='use wandb to record log')

    args = parser.parse_args()

    return args

def train(data_loader, model, device, epoch, optimizer, checkpoint_path):

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    print("--- Using Optimizer ---\n", optimizer)
    print("--- Data augmentation ---\n", data_loader.dataset.transform)

    model.train()
    training_time = []
    with profile(activities=[ProfilerActivity.CPU],
                 schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(train_config['log_path']),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True
                # on_trace_ready=trace_handler
        ) as prof:
        with torch.profiler.record_function("backward"):
            for i in range(epoch):
                print(
                    f"Epoch: {i} ----------------- \nBatch_size: {data_loader.batch_size}")
                limited_iters = 32
                for images, target in metric_logger.log_every(data_loader, int(data_loader.batch_size), header):
                    # start_time = time.perf_counter()
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    # with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    # end_time = time.perf_counter()
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    batch_size = images.shape[0]
                    # batch_training_time = end_time - start_time
                    # training_time.append(batch_training_time / batch_size)
                    # print(
                    #     f"Training Time / samples: {batch_training_time / batch_size} sec, Avg Training Time = {np.mean(training_time)} sec")

                    metric_logger.update(loss=loss.item())
                    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                    
                    limited_iters -= 1
                    prof.step()
                    print('limited_iters: ', limited_iters)
                    
                    if limited_iters == 0:
                        break
                
                # gather the stats from all processes
                metric_logger.synchronize_between_processes()
                print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                    .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
                # torch.save({
                #     'epoch': i,
                #     'model': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'loss': loss
                # },
                #     checkpoint_path)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=10))
        prof.export_stacks(train_config['log_path'] + '/profile_stack.txt', "self_cpu_time_total")
        # prof.export_chrome_trace(train_config['log_path'] + "/trace.json")
        

# def prof_model(pt_model, device):
#     def trace_handler(p):
#         # 'self' memory corresponds to the memory allocated (released) by the operator, 
#         # excluding the children calls to the other operators.
#         print(p.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
#     train(train_config['data_loader'], train_config['model'], train_config['device'], 
#             train_config['epoch'], train_config['optimizer'], train_config['checkpoint_path'])
        
#         # trace_handler(prof)
        

def main(args, config):

    model = timm.create_model('deit_small_patch16_224', pretrained=True)
    model.to('cpu')
    
    print(model)
    print(train_config['log_path'])
    
    # Load dataset
    train_dataset = datasets.ImageFolder(
    root=os.path.join(args.data_path, 'val'),
    transform=transforms.Compose([
        transforms.Resize(int(224/0.9),interpolation = 3),
        transforms.CenterCrop((224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                            std=IMAGENET_DEFAULT_STD)
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    # Set training param
    train_config['data_loader'] = train_loader
    train_config['model'] = model
    train_config['device'] = 'cpu'
    train_config['epoch'] = 1
    train_config['optimizer'] = torch.optim.SGD(model.parameters(), lr=0.001)
    train_config['checkpoint_path'] = None
    

    train(train_config['data_loader'], train_config['model'], train_config['device'], 
            train_config['epoch'], train_config['optimizer'], train_config['checkpoint_path'])
    

if __name__ == '__main__':
    args = parse_option()
    main(args, config=train_config)
