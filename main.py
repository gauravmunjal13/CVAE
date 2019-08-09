import os
import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

from argparse import ArgumentParser

import torch 
from torchsummary import summary
from tqdm import tqdm

from model.cvae import CVAE
from model.data_loader import MovieLensDataLoader
import train
import evaluate
import utils

if __name__ == '__main__':
    parser = ArgumentParser()
    # possible options are train or evaluate
    parser.add_argument("--exp_type", dest="exp_type", type=str, default="evaluate")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=30)
    # "best": for test evaluation or "last": while training
    parser.add_argument("--restore_file", dest="restore_file", type=str, default="best") 
    args = parser.parse_args()

    # set-up the configuration parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    data_dir = "/afs/inf.ed.ac.uk/user/s18/s1890219/Thesis/MovieLens/data/ml-latest-small/processed_data/pop_movies/"

    ##### load the datasets #####
    print("data_dir", data_dir)
    # first get the mapping dicts
    user_pickle_file = os.path.join(data_dir, 'user2index.pickle')
    item_pickle_file = os.path.join(data_dir, 'item2index.pickle')

    with open(user_pickle_file, 'rb') as handle:
        user2index = pickle.load(handle)
    with open(item_pickle_file, 'rb') as handle:
        item2index = pickle.load(handle)

    num_users = len(user2index)
    num_items = len(item2index)
    print("num_users", num_users)
    print("num_items", num_items)

    # load the original complete dataset
    original_data = pd.read_csv(os.path.join(data_dir, 'ratings_indexed.csv'))
    # 1 represents ratings 3.5 or greater
    print("original_data.label.value_counts()", original_data.label.value_counts())
    
    # all labels are 1 here
    train_pd = pd.read_csv(os.path.join(data_dir, 'train_plays.csv'))
    validation_pd = pd.read_csv(os.path.join(data_dir, 'val_plays.csv'))
    test_pd = pd.read_csv(os.path.join(data_dir, 'test_plays.csv'))
    
    # train user-item interaction for getting the user representation \
    # as well as for cold start users:
    train_user_item_interaction_dict = defaultdict(list)
    train_user_item_interaction = train_pd[train_pd["label"]==1].groupby("userId")["movieId"].apply(list)
    train_user_item_interaction_dict = train_user_item_interaction.to_dict(OrderedDict)
    
    # for validation precision
    val_user_item_interaction_dict = defaultdict(list)
    val_user_item_interaction = validation_pd[validation_pd["label"]==1].groupby("userId")["movieId"].apply(list)
    val_user_item_interaction_dict = val_user_item_interaction.to_dict(OrderedDict)
    
    # for test precision
    test_user_item_interaction_dict = defaultdict(list)
    test_user_item_interaction = test_pd[test_pd["label"]==1].groupby("userId")["movieId"].apply(list)
    test_user_item_interaction_dict = test_user_item_interaction.to_dict(OrderedDict)
    
    # load slates for training and validation set
    data_config = {
            "num_items": num_items,
            "data_dir": data_dir,
            "batch_size":  128,
            "train_user_item_interaction_dict": train_user_item_interaction_dict 
    }
    data_config["file_name"] = "training_slate_plays_slate_5.csv"
    data_config["data_type"] = "train"
    train_dataloader_cls = MovieLensDataLoader(data_config)
    train_dataloader = train_dataloader_cls.data_loader
    
    # DO HERE: DOWNLOAD VALIDATION SLATES
    # TEST DOESN'T REQUIRE TO BE IN SLATES    
    # user representation for validation users also remains same
    data_config["file_name"] = "val_slate_plays_slate_5.csv"
    data_config["data_type"] = "valid"
    val_dataloader_cls = MovieLensDataLoader(data_config)
    val_dataloader = val_dataloader_cls.data_loader
    
    ##### config for the model initialization #####
    model_config = {
    "num_users": num_users,
    "num_items": num_items,
    "embedding_size": 8,
    "slate_size": 10, # FOR DIFFERENT SLATE SIZES
    "hidden_dim": 128,
    "latent_dim": 16,
    "response_dim": 11, # FOR DIFFERENT SLATE SIZES
    "device": device,
    "model_dir": "./experiments/cvae/",
    "batch_size":  128
    }
    
    if not os.path.exists(model_config["model_dir"]):
        os.mkdir(model_config["model_dir"])
        
    # create folder for the logs of the experiment
    exp_logs = os.path.join(model_config["model_dir"], "output_logs/analysis7/")
    if not os.path.exists(exp_logs):
        os.mkdir(exp_logs)
        
    # create folder to save the model
    exp_saved_models = os.path.join(model_config["model_dir"], "saved_models/analysis7/")
    if not os.path.exists(exp_saved_models):
        os.mkdir(exp_saved_models)
        
    model_config["exp_logs_dir"] = exp_logs
    model_config["exp_save_models_dir"] = exp_saved_models
    
    # file path for gradient checking and plotting
    #file_loc = "/afs/inf.ed.ac.uk/user/s18/s1890219/Thesis/CVAE/experiments/cvae/output_logs/analysis_1/"
    #model_config["file_loc"] = file_loc
    # the parameter name suggest what to evaluate on
    model_config["test_user_item_interaction_dict"] = val_user_item_interaction_dict
    model_config["train_user_item_interaction_dict"] = train_user_item_interaction_dict
    
    ##### define the model #####
    model_cvae = CVAE(config=model_config).to(device)
    print(model_cvae)
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    # size_average is set to False, the losses are instead summed for each minibatch
    #criterion.size_average = False
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model_cvae.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    '''for name, param in model_cvae.named_parameters():
            if name in ["items_embedding.weight", "response_embedding.weight",
                          "f_prior.linear1.weight", "f_prior.prior_mean.weight",
                          "f_prior.prior_log_var.weight", "encoder.linear1.weight",
                          "encoder.mu.weight", "'encoder.log_var.weight'",
                          "decoder.linear1.weight", "decoder.linear3.weight"]:
                print(name, param[0])'''
    
    # train the model
    if args.exp_type == "train":
        train.train_and_val(model_cvae, train_dataloader, val_dataloader, \
                      criterion, optimizer, args, model_config)

                
    if args.exp_type == "evaluate":
        eval_config = {
            "num_items": num_items,
            "train_user_item_interaction_dict": train_user_item_interaction_dict,
            "test_user_item_interaction_dict": test_user_item_interaction_dict,
            "exp_save_models_dir": "./experiments/cvae/saved_models/analysis7/",
            "slate_size": 10 # for different slate size
        }
        # Is there a better way to wrap large num of arguments
        precision, recall, user_test_metric = evaluate.evaluation(model_cvae, args, eval_config)
        
        user_test_metric_json_file = os.path.join(eval_config["exp_save_models_dir"], "user_test_metric.json")
        utils.save_dict_to_json(user_test_metric, user_test_metric_json_file)
        
        
   
    
    