import os
import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from prettytable import PrettyTable


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1,3,1,1)
    stds = torch.tensor(stds).reshape(1,3,1,1)
    return images*stds+means

def show_batch(batches, nums, idx, stats):
    assert 0<=idx<len(batches), 'invalid batch number'
    rows = round(math.sqrt(nums))
    for i, (images, labels) in enumerate(batches):
        if i != idx:
            continue
        assert 0<nums<=len(images), 'not enough images in batch'
        print(len(images))
        fig, ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([])
        ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        ax.imshow(make_grid(denorm_images[:nums], nrow=rows).permute(1,2,0).clamp(0,1))
        break

def count_parameters(model, print_table=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    if print_table:
        print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
        
def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==targets).item()/len(preds))

def validate(model, batch):
    images, targets = batch
    outputs = model(images)
    acc = accuracy(outputs, targets)
    return acc