# -*- coding: utf-8 -*-
import os
import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

from argparse import ArgumentParser

import torch 
from torch.autograd import Variable
from torchsummary import summary
from tqdm import tqdm

import utils


def compute_loss(criterion, reconstructed_response_reshaped, response, 
                 z_mean, z_log_var, batch_size):
    '''
    '''
    annealing = 0.5
    reconstruction_loss = criterion(reconstructed_response_reshaped, response) / batch_size
    KL_loss = torch.mean(annealing*torch.sum( 
            torch.exp(z_log_var) + z_mean**2 -1 - z_log_var, 1) )
    return reconstruction_loss + KL_loss
    
    

def train(model, data_loader, criterion, optimizer, config, epoch):
    '''
    return the total epoch losses and every 2000 batches
    '''
    model.train()
    # running loss to capture every 2000 batches
    running_loss = 0.0
    # save the running loss
    epoch_losses = []
    # total loss for an epoch
    total_epoch_loss = 0.0

    #with tqdm(total=len(data_loader.dataset)) as t:
    for i, data in enumerate(data_loader):
        # get the inputs
        userId, user_repr, slate, response, response_encoded = (data[0].to(config["device"]),
                                   data[1].to(config["device"]),
                                   data[2].to(config["device"]),
                                   data[3].to(config["device"]),
                                   data[4].to(config["device"]))
        
        # is wrapping into variable type required?
        userId, user_repr, slate, response, response_encoded = Variable(userId), Variable(user_repr), \
                                     Variable(slate), Variable(response), Variable(response_encoded)
        
        # forward pass
        z_mean, z_log_var, reconstructed_slate, reconstructed_response_reshaped = model(user_repr, \
                                                                                slate, response_encoded)
        
        # compute the loss
        loss = compute_loss(criterion, reconstructed_response_reshaped, 
                            response, z_mean, z_log_var, config["batch_size"])
        
        # zero the gradients
        optimizer.zero_grad()
        
        # do the backward pass
        loss.backward()
        
        # update the weights
        optimizer.step()
        
        # save statistics
        running_loss += loss.item()
        total_epoch_loss += loss.item()
        
        if i % 20 == (20-1):    # every 2000 mini-batches
            print('[%d, %5d] training loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
            epoch_losses.append(running_loss)
            running_loss = 0.0
        
        ## update the average loss
        #t.set_postfix(loss='{:05.3f}'.format(total_epoch_loss/(i+1)))
        #t.update()
        
    return np.mean(total_epoch_loss/(i+1)), epoch_losses

def validation(model, data_loader, criterion, config, epoch):
    '''
    return the total epoch losses and every 2000 batches
    '''
    model.eval()

    # running loss to capture every 2000 batches
    running_loss = 0.0
    # save the running loss
    epoch_losses = []
    # total loss for an epoch
    total_epoch_loss = 0.0
    #with tqdm(total=len(data_loader.dataset)) as t:
    for i, data in enumerate(data_loader):
        # get the inputs
        userId, user_repr, slate, response, response_encoded = (data[0].to(config["device"]),
                                   data[1].to(config["device"]),
                                   data[2].to(config["device"]),
                                   data[3].to(config["device"]),
                                   data[4].to(config["device"]))
        
        # is wrapping into variable type required?
        userId, user_repr, slate, response, response_encoded = Variable(userId), Variable(user_repr), \
                                     Variable(slate), Variable(response), Variable(response_encoded)
        
        # forward pass
        z_mean, z_log_var, reconstructed_slate, reconstructed_response_reshaped = model(user_repr, \
                                                                                slate, response_encoded)
        
        # compute the loss
        loss = compute_loss(criterion, reconstructed_response_reshaped, 
                            response, z_mean, z_log_var, config["batch_size"])
        
        # print statistics
        running_loss += loss.item()
        total_epoch_loss += loss.item()
        if i % 5 == (5-1):    # every 2000 mini-batches
            # save this somewhere
            print('[%d, %5d] validation loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
            epoch_losses.append(running_loss)
            running_loss = 0.0
        
        ## update the average loss
        #t.set_postfix(loss='{:05.3f}'.format(total_epoch_loss/(i+1)))
        #t.update()

    return np.mean(total_epoch_loss/(i+1)), epoch_losses


def train_and_val(model, train_dataloader, val_dataloader, criterion, optimizer, args, config):
    '''
    input:
        model: torch.nn.Module, the neural network
        train_loader: a torch.utils.data.DataLoader obj for the training
        val_dataloader: for validation
        criterion: to compute the loss function
        args: parameters from the command line
        config: parameters specified inside
        # no metrics is used during training and validation
        # while loss is useless for testing.
    '''
    # restore weights here if starting in middle? PENDING

    print("len(train_dataloader.dataset)", len(train_dataloader.dataset))
    print("len(val_dataloader.dataset)", len(val_dataloader.dataset))

    epoch_loss = {"train_loss": [], "val_loss": []}
    loss_batches = {"train_loss": [], "val_loss": []}
    best_val_loss = 1.0

    for epoch in range(args.num_epochs):
        # run one epoch
        print("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        # need to get the loss from the training too
        # train_loss for each epoch
        train_epoch_loss, train_loss_batches = train(model, train_dataloader, 
                                                     criterion, optimizer, config, epoch)
        epoch_loss["train_loss"].append(train_epoch_loss)
        loss_batches["train_loss"].append(train_loss_batches)
        # when to save the model is still not clear: during training the optimizer \
        # will change the weights of the model. Seems like save the latest weights \
        # and the best one
        
        val_loss, val_loss_batches = validation(model, val_dataloader, criterion, config, epoch)
        epoch_loss["val_loss"].append(val_loss)
        loss_batches["val_loss"].append(val_loss_batches)

        # check for the best model
        is_best = val_loss<=best_val_loss
        
        # Save weights, overwrite the last one and the new best one
        print("save_checkpoint")
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=is_best,
                               checkpoint=config["exp_save_models_dir"])

        if is_best:
            print("is_best")
            best_loss_json_file = os.path.join(config["exp_save_models_dir"], "best_loss_batches.json")
            utils.save_dict_to_json(loss_batches, best_loss_json_file)

        # overwrite the last epoch losses
        last_loss_json_file = os.path.join(config["exp_save_models_dir"], "last_loss_batches.json")
        utils.save_dict_to_json(loss_batches, last_loss_json_file)

    all_epoch_loss_json_file = os.path.join(config["exp_save_models_dir"], "all_epoch_loss.json")
    utils.save_dict_to_json(epoch_loss, all_epoch_loss_json_file)
