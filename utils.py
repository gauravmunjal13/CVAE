import json
import os
import shutil

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# reference: cs230-standford code examples
def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

# reference: cs230-standford code examples
def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

# reference: cs230-standford code examples
def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k:v for k, v in d.items()}
        json.dump(d, f, indent=4)
    
# reference: https://discuss.pytorch.org
#https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html
def plot_grad_flow(named_parameters, file_name = None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for name, param in named_parameters:
        if param.grad is None:
            continue
        if(param.requires_grad) and ("bias" not in name):
            layers.append(name)
            ave_grads.append(param.grad.abs().mean())
            max_grads.append(param.grad.abs().max())    
    
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    #plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    #plt.legend([Line2D([0], [0], color="c", lw=4),
    #            Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])
    #plt.legend([Line2D([0], [0], color="b", lw=4)],['mean-gradient'])
    
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=1200)
        
def plot_loss_stats(train_val_loss, loss_type = "Loss", file_name = None):
    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    
    ax_1.plot(np.arange(len(train_val_loss["train_loss"]))+1, 
              train_val_loss["train_loss"], label="Train loss")
    ax_1.plot(np.arange(len(train_val_loss["train_loss"]))+1,
              train_val_loss["val_loss"], label="Val loss")
    
    ax_1.legend(loc=0)
    ax_1.set_title(loss_type)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel("Loss")
    #plt.xticks(rotation="vertical")
    
    if file_name is not None:
        fig_1.savefig(file_name, bbox_inches='tight', dpi=1200)
    
