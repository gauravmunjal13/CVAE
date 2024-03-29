# -*- coding: utf-8 -*-
import os
from collections import OrderedDict, defaultdict
from argparse import ArgumentParser
import numpy as np

import torch 
from torch.autograd import Variable
from tqdm import tqdm

import utils


def get_user_repr_eval(userId, config):
    '''
    input: user: userId
    output: returns a list
    The implementation is more simples than used in training \
    since only one user is tested at a time.
    '''
    train_user_item_interaction = config["train_user_item_interaction_dict"][userId]
    #user_repr = np.zeros(config["num_items"], dtype=int)
    #user_repr[train_user_item_interaction] = 1
    return train_user_item_interaction

def evaluation(model, args, config):
    '''
    evaluate the precision of the model by computing the average across users
    input:
        model: torch.nn.Module, the neural network
        config:
            num_items:
            train_user_item_interaction_dict: to create the user_repr
            test_user_item_interaction_dict: for precision cal
            exp_save_models_dir: to load the model from this place
            slate_size:
    output:
        precision: prec of the model
        recall: 
    ''' 
    if args.restore_file is not None:
        print("args.restore_file", args.restore_file)
        restore_path = os.path.join(config["exp_save_models_dir"], args.restore_file + '.pth.tar')
        checkpoint = utils.load_checkpoint(restore_path, model)  
        print("Epoch no. retrieved", checkpoint["epoch"])
    
    # put the model in eval mode
    model.eval()
    
    total_items = set(np.arange(config["num_items"]))
    
    precision = defaultdict(list) # indexed by the user
    recall = defaultdict(list)
    cold_start_user_counter = 0
    prec_rec_user_dict = defaultdict(dict)
    
    with torch.no_grad(): # what is the use of this?
    # do for each user
        for test_user, test_user_interactions in config['test_user_item_interaction_dict'].items():
            
            # for each user, identify all the movies he hasn't interacted before
            if test_user in config['train_user_item_interaction_dict']:
                input_items = list(total_items.difference(
                        config["train_user_item_interaction_dict"][test_user]))
            else:
                # cold-start case: No recommendation for this user
                cold_start_user_counter += 1
                continue
            
            # get the user representation from the training interactions
            # tensor correponding to training interactions, i.e. indices
            user_repr = torch.tensor(get_user_repr_eval(test_user, config))
            
            # optimal response is getting max response
            #optimal_response_encoded = torch.Tensor([0,0,0,0,0,0,0,0,0,1,0])
            optimal_response = torch.tensor([0])
            
            recommended_slate = model.inference( 
                                        user_repr, optimal_response, input_items, config)
            
            if test_user == 5:
                print(recommended_slate)
            
            relevant = set(config['test_user_item_interaction_dict'][test_user])
            
            num = len(relevant.intersection(recommended_slate))
            prec_den = len(recommended_slate)
            precision[test_user] = num/prec_den
            
            rec_den = len(relevant)
            recall[test_user] = num/rec_den
            
            #print("test_user: precision[test_user]", test_user, precision[test_user])
            #print("test_user: recall[test_user]", test_user, recall[test_user])
            
            prec_rec_user_dict[test_user] = {"prec": num/prec_den, "rec": num/rec_den, 
                    "num_train_interactions": len(config["train_user_item_interaction_dict"][test_user]),
                    "num_test_interactions": len(config["test_user_item_interaction_dict"][test_user])}
        
        Precision = sum(prec for prec in precision.values())/len(precision)
        Recall = sum(rec for rec in recall.values())/len(recall)
        
        #if args.exp_type == "evaluate":
        local_prec = 0
        local_rec = 0
        local_counter = 0
        active_users_prec = 0
        active_users_prec_counter = 0
        for key, item in prec_rec_user_dict.items():
            if item["num_test_interactions"] >= config["slate_size"]:
                local_prec += item["prec"]
                #local_rec += item["rec"]
                local_counter += 1
                # precision for highly active users
                if item["num_train_interactions"] > 30:
                    active_users_prec += item["prec"]
                    active_users_prec_counter += 1
        print("Precision:", local_prec/local_counter)
        print("Precision for active users:", active_users_prec/active_users_prec_counter)
        #print("Local rec:", local_rec/local_counter)
            
        
        #print("Precision: ", sum(prec for prec in precision.values())/len(precision))
        print("Recall: ", sum(rec for rec in recall.values())/len(recall))
        print("cold_start_user_counter", cold_start_user_counter)
        
        return Precision, Recall, prec_rec_user_dict
        