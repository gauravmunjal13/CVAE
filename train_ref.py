import os
from collections import OrderedDict, defaultdict
from argparse import ArgumentParser
import numpy as np

import torch 
from torch.autograd import Variable
from tqdm import tqdm

from model.matrix_factorization import MatrixFactorization
from model.data_loader import MovieLensDataLoader
import utils

# ref: https://github.com/beckernick/matrix_factorization_recommenders/blob/master/matrix_factorization_recommender.ipynb
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
        inputs, labels = data[0].to(config["device"]), data[1].to(config["device"])
        
        # is wrapping into variable type required?
        inputs, labels = Variable(inputs), Variable(labels)
        
        # forward pass
        y_pred = model(inputs)
        
        # compute the loss
        loss = criterion(y_pred, labels)
        
        # zero the gradients
        optimizer.zero_grad()
        
        # do the backward pass
        loss.backward()
        
        # update the weights
        optimizer.step()
        
        # save statistics
        running_loss += loss.item()
        total_epoch_loss += loss.item()
        
        if i % 2000 == (2000-1):    # every 2000 mini-batches
            print('[%d, %5d] training loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            epoch_losses.append(running_loss)
            running_loss = 0.0
        
        ## update the average loss
        #t.set_postfix(loss='{:05.3f}'.format(total_epoch_loss/(i+1)))
        #t.update()
    print(i)
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
        inputs, labels = data[0].to(config["device"]), data[1].to(config["device"])
        
        # is wrapping into variable type required?
        inputs, labels = Variable(inputs), Variable(labels)
        
        # forward pass
        y_pred = model(inputs)
        
        # compute the loss
        loss = criterion(y_pred, labels)
        
        # do the backward pass
        loss.backward()
        
        # print statistics
        running_loss += loss.item()
        total_epoch_loss += loss.item()
        
        if i % 2000 == (2000-1):    # every 2000 mini-batches
            # save this somewhere
            print('[%d, %5d] validation loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
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
    print("train_dataloader.dataset.train_data.shape[0]", train_dataloader.dataset.train_data.shape[0])

    total_loss = {"train_loss": [], "val_loss": []}
    loss_batches = {"train_loss": [], "val_loss": []}
    best_val_loss = 100.0

    for epoch in range(args.num_epochs):
        # run one epoch
        print("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        # need to get the loss from the training too
        # train_loss for each epoch
        train_loss, train_loss_batches = train(model, train_dataloader, criterion, optimizer, config, epoch)
        total_loss["train_loss"].append(train_loss)
        loss_batches["train_loss"].append(train_loss_batches)
        # when to save the model is still not clear: during training the optimizer \
        # will change the weights of the model. Seems like save the latest weights \
        # and the best one
        
        val_loss, val_loss_batches = validation(model, val_dataloader, criterion, config, epoch)
        total_loss["val_loss"].append(val_loss)
        loss_batches["val_loss"].append(val_loss_batches)
        
        print("train_loss, val_loss", train_loss, val_loss)

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
            best_loss_json_file = os.path.join(config["exp_save_models_dir"], "best_epoch_losses.json")
            utils.save_dict_to_json(loss_batches, best_loss_json_file)

        # overwrite the last epoch losses
        last_loss_json_file = os.path.join(config["exp_save_models_dir"], "last_epoch_losses.json")
        utils.save_dict_to_json(loss_batches, last_loss_json_file)

    all_epoch_loss_json_file = os.path.join(config["exp_save_models_dir"], "all_epoch_loss.json")
    utils.save_dict_to_json(total_loss, all_epoch_loss_json_file)

def compute_precision(user_id, top_k_items, relevant_user_item_dict):
    '''
    Compute precision for the given user_id. 
    inputs: 
    user_id: a single test user id
    top_k_items: list of top k recommended items for the user_id
    relevant_user_item_dict: dict indexed by test user and 
        values as the list of movies he liked
    output:
    prec: precision value for that given user_id.'''
    # user 2084 didn't like any movie in the test set, question here
    if user_id in relevant_user_item_dict: 
        relevant_items = relevant_user_item_dict[user_id]
    else: 
        # not possible in the new case with implicit feedback
        relevant_items = [] # empty list
    num = len(list(set(relevant_items) & set(top_k_items)))
    den = len(top_k_items)
    prec = num/den
    
    return prec


def evaluation(model, test_users, total_items, train_user_item_dict, 
               relevant_user_item_dict, args, config):
    '''
    evaluate the precision of the model by computing the average across users
    input:
        model: torch.nn.Module, the neural network
    output:
        precision: prec of the model
    ''' 
    if args.restore_file is not None:
        print(args.restore_file)
        restore_path = os.path.join(config["exp_save_models_dir"], args.restore_file + '.pth.tar')
        utils.load_checkpoint(restore_path, model)   
    
    # put the model in eval mode
    model.eval()

    precisions = defaultdict(list) # indexed by the user
    
    cold_start_user_counter = 0
    with torch.no_grad(): # what is the use of this?
    # do for each user
        for test_user in test_users:
            print('test_user', test_user)
            # for each user, identify all the movies he hasn't interacted before
            if test_user in train_user_item_dict:
                input_items = list(total_items.difference(train_user_item_dict[test_user]))
            else:
                # cold-start case: No recommendation for this user
                input_items = list(total_items)
                cold_start_user_counter += 1
                #continue
            
            # create the batch, for that user, as a user-movie pair
            test_user_torch = torch.full((len(input_items),), test_user, dtype= torch.long, device=device).unsqueeze(dim=1)
            input_items_torch = torch.tensor(input_items, dtype= torch.long, device=device).unsqueeze(dim=1)

            #print("train_user_item_dict[test_user]", train_user_item_dict[test_user])
            print("len(input_items)", len(input_items))
            
            inputs = torch.cat((test_user_torch, input_items_torch), dim=1)
            # here, have to send to the device as the model is also running on device
            inputs = inputs.to(device)
            
            # forward pass
            # y_pred is the tensor availiable at device, hence may be required to detach
            y_pred = model(inputs)
            
            # can't compute the test loss compute the loss as there is no label for the uninteracted items
            
            #print("y_pred.shape", y_pred.shape)
            #print("y_pred[:3]", y_pred[:3])
            
            # the batch corresponds to a paritcular user, the y_pred is the prediction across all possible items
            # get the top k predictions, results in tensor of shape 3 from the batch size, let's say [123]
            _, top_k_items_index = torch.topk(y_pred, 10)
            print("top_k_items_index", top_k_items_index)
            # need to match the top_k_items back to the moviesId: PENDING is mapping
            # need to see what's on GPU or CPU: Google
            #print("top_k", top_k_items_index)
            #print("type(top_k_items_index)", type(top_k_items_index))
            
            # now the item's indices are same as item's IDs
            #top_k_items = [index2item[int(index)] for index in top_k_items_index]

            # compute the precision for each user
            precisions[test_user] = compute_precision(test_user, top_k_items_index, relevant_user_item_dict)

            print("test_user: precisions[test_user]", test_user, precisions[test_user])
            
        print("Precision: ",sum(prec for prec in precisions.values())/len(precisions))

    print("cold_start_user_counter", cold_start_user_counter)

if __name__ == '__main__':
    parser = ArgumentParser()
    # possible options are train or evaluate
    parser.add_argument("--exp_type", dest="exp_type", type=str, default="train")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=30)
    # best or last
    parser.add_argument("--restore_file", dest="restore_file", type=str, default="last") 
    args = parser.parse_args()

    # set-up the configuration parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    data_dir = "./data/ml-latest-small/processed_data/"

    ##### load the datasets #####
    data_config = {
        "data_dir": data_dir,
        "file_name": "train.csv",
        "batch_size":  64,
        "data_type": "train"
        } 
    train_dataloader_cls = MovieLensDataLoader(data_config)
    train_dataloader = train_dataloader_cls.data_loader
    data_config["file_name"] = "validation.csv"
    data_config["data_type"] = "valid"
    val_dataloader_cls = MovieLensDataLoader(data_config)
    val_dataloader = val_dataloader_cls.data_loader
    data_config["file_name"] = "test.csv"
    data_config["data_type"] = "test"
    test_dataloader_cls = MovieLensDataLoader(data_config)
    test_dataloader = test_dataloader_cls.data_loader

    ##### config for the model initialization #####
    num_users = train_dataloader_cls.num_users + val_dataloader_cls.num_users + test_dataloader_cls.num_users
    num_items = train_dataloader_cls.num_items + val_dataloader_cls.num_items + test_dataloader_cls.num_items
    print("num_users", num_users)
    print("num_items", num_items)

    model_config = {
    "num_users": num_users,
    "num_items": num_items,
    "embedding_size": 40,
    "map_dir": "./data/data/ml-latest-small/processed_data/",
    "map_user_file": "user2index.pickle",
    "map_item_file": "item2index.pickle",
    "device": device,
    "model_dir": "./experiments/matrix_factorization/" 
    }

    if not os.path.exists(model_config["model_dir"]):
        os.mkdir(model_config["model_dir"])

    # create folder for the logs of the experiment
    exp_logs = os.path.join(model_config["model_dir"], "output_logs")
    if not os.path.exists(exp_logs):
        os.mkdir(exp_logs)

    # create folder to save the model
    exp_saved_models = os.path.join(model_config["model_dir"], "saved_models")
    if not os.path.exists(exp_saved_models):
        os.mkdir(exp_saved_models)

    model_config["exp_logs_dir"] = exp_logs
    model_config["exp_save_models_dir"] = exp_saved_models

    ##### define the model #####
    model_MF = MatrixFactorization(config=model_config).to(device)
    # Or binary cross entropy loss
    criterion = torch.nn.BCELoss()
    learning_rate = 0.1 #1e-6
    optimizer = torch.optim.SGD(model_MF.parameters(), lr=learning_rate) 

    # train the model
    if args.exp_type == "train":
        train_and_val(model_MF, train_dataloader, val_dataloader, criterion, optimizer, args, model_config)

    ##### evaluation on the test data #####

    if args.exp_type == "evaluate":
        # setting-up the configuration
        
        # reverese mapping: No need now, as the ID's were replaced with indices in preprocessing
        #index2user = {v: k for k, v in model_MF.user2index.items()}
        #index2item = {v: k for k, v in model_MF.item2index.items()}

        # test_users: list of all users in test set which liked the movie
        test_users = test_dataloader_cls.users
        print("len(test_users)", len(test_users))
        
        # total_items: set of all possible movie Ids
        # the items attribute of the data loader class is set
        total_items = train_dataloader_cls.items | val_dataloader_cls.items | test_dataloader_cls.items
        print("len(total_items)", len(total_items))

        # get the user-item interactions in the training set
        
        # train_user_item_dict: dict indexed by user and set of movies interacted in the training data
        train_user_item = train_dataloader_cls.data_pd[train_dataloader_cls.data_pd["label"]==1] \
                                              .groupby("userId")["movieId"].apply(set)
        train_user_item_dict = train_user_item.to_dict(OrderedDict)
        
        # relevant_user_item_dict: dict indexed by test user and \
        # values as the list of the movies interacted
        relevant_items = test_dataloader_cls.data_pd[test_dataloader_cls.data_pd["label"]==1]
        relevant_user_item = relevant_items.groupby("userId")["movieId"].apply(list)
        relevant_user_item_dict = relevant_user_item.to_dict(OrderedDict)

        eval_config = {
            "test_users": test_users,
            "total_items": total_items,
            "train_user_item_dict": train_user_item_dict,
            "relevant_user_item_dict": relevant_user_item_dict,
            "exp_save_models_dir": "./experiments/matrix_factorization/saved_models/",
        }
        # Is there a better way to wrap large num of arguments
        evaluation(model_MF, test_users, total_items, train_user_item_dict, 
                   relevant_user_item_dict, args, eval_config)

    print("Finished")

