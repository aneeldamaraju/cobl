from contextlib import contextmanager
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from einops import rearrange, repeat

from cobl.datasets import reverse_transform

def split_layers(x0):
    return rearrange(x0, "b v h w (n c) -> b v n h w c", c=3)  # [B, V, 7, 3, 512, 512]

def count_obj_layers(sampler):
    count = 0
    for mask in sampler.logger.masks[-1][:-1]:
        if np.mean(mask > 0.5) > 0.02: 
            count += 1
    return count

def plot_obj_layers(scene, sampler, out):
    nlayers = count_obj_layers(sampler)
    out_imgs = split_layers(out).squeeze()
    total_layers = int(out_imgs.shape[-1]/3)
    fig, ax = plt.subplots(1,nlayers+2, figsize=(3*nlayers+2, 3))
    ax[0].imshow(scene)
    ax[0].axis('off')
    ax[0].set_title('Image')
    
    for i,l in enumerate(reversed(range(total_layers-nlayers-1,total_layers))):
        ax[i+1].imshow(out_imgs[l-1])
        ax[i+1].axis('off')
    ax[1].set_title('Background Layer')
    ax[-1].set_title('Foreground Layer')
    return fig