import os
import ast
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    '''
    define the custom dataset to be used in data loader
    '''
    def __init__(self, data_pd ,transforms=None):
        # let's return separately for each of the columns
        # data_pd["userId] is an int
        self.userId = torch.LongTensor(data_pd['userId'].values)
        self.user_repr = torch.tensor(data_pd['user_repr'].values.tolist())
        # data_pd["slate"] is a string of list of slate items
        self.slate = torch.tensor(data_pd['slate'].apply(ast.literal_eval))
        self.response = torch.tensor(data_pd['response'].apply(ast.literal_eval), dtype=torch.float32)
        self.response_label = torch.tensor(data_pd['response_label'].values)
        # torch.Tensor instead of torch.tensor will be userful in cat with user_embedding later on
        #self.response_encoded = torch.Tensor(data_pd['response_encoded'].apply(ast.literal_eval))
        self.transforms = transforms
        
    def __getitem__(self, index):
        userId = self.userId[index]
        user_repr = self.user_repr[index]
        slate = self.slate[index]
        response = self.response[index]
        response_label = self.response_label[index]
        #response_encoded = self.response_encoded[index]
        # the type of return is single entity: is it correct?
        #return (userId, user_repr, slate, response, response_label, response_encoded)
        return (userId, user_repr, slate, response, response_label)

    def __len__(self):
        count = self.userId.shape[0]
        return count
    
class MovieLensDataLoader:
    '''
    input: 
    config: a dict of parameters
     num_items: 
     data_dir: path of the data
     batch_size:
     train_user_item_interaction_dict:
     file_name: should be a csv file
     data_type: train, valid or test
    '''    
    
    def __init__(self, config):
        # save the config
        # I am not explicitly storing the data type as it is saved in the self.config
        self.config = config
        
        file_name = os.path.join(self.config['data_dir'], self.config['file_name'])
        # I assume the file is a csv file
        self.data_pd = pd.read_csv(file_name)
        
        # problem of cold start user
        if self.config["data_type"] == "valid":
            train_users_list = self.config["train_user_item_interaction_dict"].keys()
            # remove cold start validation users
            indices = self.data_pd["userId"].isin(train_users_list)
            #print("indices.value_counts()", indices.value_counts())
            self.data_pd = self.data_pd[indices].reset_index()
            
        self.data_pd["user_repr"] = self.data_pd["userId"].map(lambda x: self._get_user_repr(x))
        
        # create the data loader
        self.dataset = CustomDataset(self.data_pd)
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.config['batch_size'],
                                      shuffle=True, num_workers=4)
        
    def _get_user_repr(self, userId):
        '''
        input: user: userId
        '''
        train_user_item_interaction = self.config["train_user_item_interaction_dict"][userId]
        user_repr = np.zeros(self.config["num_items"], dtype=np.int64)
        user_repr[train_user_item_interaction] = 1
        # converting to list from numpy array: Must not be done
        return user_repr
    
    
       

