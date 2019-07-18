################################################################
# Load the libraries
################################################################
import os
import pickle
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
################################################################
# Define the model
################################################################

class Encoder(nn.Module):
    ''' 
    Encoder class
    '''
    def __init__(self, config):
        '''
         input:
         config: a dict of parameters
            num_users: 
            num_items:
            embedding_size:
            slate_size:
            hidden_dim: dim of the hidden layer
            latent_dim: dim of the mu and log_var
            response_dim:  dimension of one-hot representation of the conditioning vector
            device:
            model_dir:
        '''
        super().__init__()

        # save the cofig
        self.config = config

        # input_dim: input dimension to the encoder linear layer
        input_dim = config["slate_size"]*config["embedding_size"]
        # condition_dim: sum of user embedding and the one-hot representation of the response
        condition_dim = config["embedding_size"] + config["response_dim"]

        self.linear = nn.Linear(input_dim + condition_dim, config["hidden_dim"])
        self.mu = nn.Linear(config["hidden_dim"], config["latent_dim"])
        self.log_var = nn.Linear(config["hidden_dim"], config["latent_dim"])

    def forward(self, x):
        '''
        input x: shape [batch_size, input_dim + condition_dim]
        '''

        # shape of hidden [batch_size, hidden_dim]
        hidden = F.relu(self.linear(x))
        
        # latent parameters:

        # shape of mean [batch_size, latent_dim]
        mean = self.mu(hidden)
       
        # shape of log_var [batch_size, latent_dim]
        log_var = self.log_var(hidden)

        return mean, log_var

class Decoder(nn.Module):
    '''
    Decoder class
    '''
    def __init__(self, config):
        '''
         input:
         config: a dict of parameters
            config: a dict of parameters
            num_users: 
            num_items:
            embedding_size:
            slate_size:
            hidden_dim: dim of the hidden layer
            latent_dim: dim of the mu and log_var
            response_dim:  dimension of one-hot representation of the conditioning vector
            device:
            model_dir:
        '''
        super().__init__()

        self.config = config

        # input_dim: input dimension to the encoder linear layer
        input_dim = config["slate_size"]*config["embedding_size"]
        # condition_dim: sum of user embedding and the one-hot representation of the response
        condition_dim = config["embedding_size"] + config["response_dim"]

        self.linear1 = nn.Linear(config["latent_dim"] + condition_dim, config["hidden_dim"])
        self.linear2 = nn.Linear(config["hidden_dim"], input_dim)

    def forward(self, z):
        '''
        shape of z [batch_size, latent_dim + condition_dim]
        '''
        # shape of hidden [batch_size, hidden_dim]
        hidden = F.relu(self.linear1(z))
        # shape of reconstructed_x [batch_size, input_dim]
        reconstructed_x = F.relu(self.linear2(hidden))

        return reconstructed_x

class CVAE(nn.Module):
    '''
    CVAE class combining encoder and decoder classes
    '''
    def __init__(self, config):
        '''
        input:
         config: a dict of parameters
            num_users: 
            num_items:
            embedding_size:
            slate_size:
            hidden_dim: dim of the hidden layer
            latent_dim: dim of the mu and log_var
            response_dim:  dimension of one-hot representation of the conditioning vector
            device:
            model_dir:
            batch_size:
        '''
        super().__init__()
        self.config = config
        #self.users_embedding = nn.Embedding(config['num_users'], config['embedding_size'])
        self.items_embedding = nn.Embedding(config['num_items'], config['embedding_size'])
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.softmax = nn.Softmax(dim=2)

    def _sample_latent(self, mean, log_var):
        '''
        '''
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mean + eps*std

    def forward(self, user_repr, slate, response_encoded):
        '''
        input:
        u: userID tensor of shape [batch_size]
        user_repr: [batch_size, num_items]
        s: slate tensor of shape [batch_size, slate_size]
        r: response tensor in one-hot vector representation [batch_size, response_dim]
        '''
        # user: [batch_size, embedding_size]
        # Padding could be an easier option
        user = torch.empty((0,16))
        for i in range(user_repr.shape[0]):
            # required a numpy array for np.where with only condition as argument, \
            # which will return indices of the 1's
            temp_array =  np.array(user_repr[i])
            # take the first element of the returned tuple, which are indices \
            # in our case movie Ids
            user_interactions = np.where(temp_array==1)[0] 
            # convert back to tensor for accessing the embeddings
            user_interactions_tensor = torch.tensor(user_interactions)
            single_user = torch.mean(self.items_embedding(user_interactions_tensor), dim=0).view(1,-1)
            user = torch.cat((user, single_user), dim=0)
        #user = torch.mean(self.items_embedding(user_repr), dim=1)
                
        # conditioning_vector of shape [batch_size, embedding_size + response_dim]
        conditioning_vector = torch.cat((user, response_encoded), dim=1)

        # slate_embedding: tensor shape [batch_size, slate_size, embedding_size]
        # CORRECT AND VERIFIED IMPLEMENTATION
        slate_embedding = self.items_embedding(slate)
        
        # for unknown batch_size
        # slate_embedding_vector: shape [batch_size, slate_size*embedding_size] 
        # CROSS-CHECK
        slate_embedding_vector = slate_embedding.view(-1, 
                                        self.config["slate_size"]*self.config["embedding_size"]) 
        

        # input_x: shape [batch_size, (slate_size*embedding_size) + (embedding_size + response_dim)]
        input_x = torch.cat((slate_embedding_vector, conditioning_vector), dim=1)
        
        z_mean, z_log_var = self.encoder(input_x)
        sampled_z = self._sample_latent(z_mean, z_log_var)
        
        # input_z: [batch_size, (latent_dim + embedding_size + response_dim)]
        input_z = torch.cat((sampled_z, conditioning_vector), dim=1)
        
        # shape [batch_size, (slate_size*embedding_size)]
        reconstructed_x = self.decoder(input_z)
        
        # reconstructed_x_reshape: shape [batch_size, slate_size, embedding_size]
        # CROSS-CHECK
        reconstructed_x_reshape = reconstructed_x.view(-1,self.config["slate_size"], self.config["embedding_size"])

        # dot-product layer
        num_items_tensor = torch.arange(self.config["num_items"])
        # all_items_embedding: [num_items, embedding_size]
        all_items_embedding = self.items_embedding(num_items_tensor)
        # dot_product: [batch_size, slate_size, num_items]
        # CORRECT AND VERIFIED IMPLEMENTATION
        dot_product = torch.einsum("bkh, nh -> bkn", reconstructed_x_reshape, all_items_embedding)
        
        # k headsoftmax layer, across the last dimension of [batch_size, slate_size, num_items] 
        softmax_layer = self.softmax(dot_product)
        
        # reconstructed_s shape: [batch_size, slate_size]
        # argmax to be done during inference only
        reconstructed_slate = torch.argmax(softmax_layer, dim=2)

        # slate_reshaped: [batch_size, slate_size, 1]
        slate_reshaped = slate.view(-1,self.config["slate_size"],1)
        
        # reconstructed_response: [batch_size, slate_size, 1]
        # Can be mentioned in implementation challanges
        # CORRECT AND VERIFIED IMPLEMENTATION
        reconstructed_response = torch.gather(softmax_layer, 2, slate_reshaped)
        
        # reconstructed_response_reshaped: [batch_size, slate_size]
        reconstructed_response_reshaped = reconstructed_response.view(-1, self.config["slate_size"])
        
        return z_mean, z_log_var, reconstructed_slate, reconstructed_response_reshaped


    def inference(self, user_repr, response_encoded, input_items):
        '''
        input:
         user_repr: tensor object for a single user interactions, \
            i.e. item indicies
         response_encoded: tensor for the optimal response
         input_items: indices of the items not interacted
        '''
        SAFE_POINT = 30 # to counter for the seen interactions
        slate_size = 10 
        # notice the dim here, this time no batch is there
        # user: [embedding_size]
        # This is a MOOT point to mention
        user = torch.mean(self.items_embedding(user_repr), dim=0)
        
        # shape: [embedding_size+response_dim]
        conditioning_vector = torch.cat((user, response_encoded), dim=0)
         
        # sample z: [embedding_size]
        z = torch.randn(self.config["latent_dim"])
        
        input_z = torch.cat((z, conditioning_vector), dim=0)
        
        # shape [(slate_size*embedding_size)]
        reconstructed_x = self.decoder(input_z)
        
        # reconstructed_x_reshape: shape [batch_size, slate_size, embedding_size]
        # CROSS-CHECK
        reconstructed_x_reshape = reconstructed_x.view(self.config["slate_size"], self.config["embedding_size"])
        
        # dot-product layer
        # MOOT point: I can't filter out the seen interactions from embeddings, \
        # as it will reduce the dimension of all_items_embedding and hence results \
        # in wrong argmax recommendation, because the indices get different
        # all_items_embedding: [num_items, embedding_size]
        num_items_tensor = torch.arange(self.config["num_items"])
        all_items_embedding = self.items_embedding(num_items_tensor)
        # dot_product: [slate_size, num_items]
        # CORRECT AND VERIFIED IMPLEMENTATION
        dot_product = torch.einsum("kh, nh -> kn", reconstructed_x_reshape, all_items_embedding)
        
        # k headsoftmax layer, across the last dimension of [slate_size, num_items] 
        softmax_layer = F.softmax(dot_product)
        
        # argmax to be done during inference probably
        #reconstructed_slate = torch.argmax(softmax_layer, dim=1)
        
        # reconstructed_slate shape: [slate_size, k=100]
        _, reconstructed_slate = torch.topk(softmax_layer,k=SAFE_POINT, dim=1)
        recommended_slate = []
        for i in range(slate_size):
            rec_item = reconstructed_slate[i].numpy()
            mask = np.isin(rec_item, input_items)
            rec_item = rec_item[mask]
            # take the top unseen item as a recommendation for that slate position
            for j in range(slate_size):
                item = rec_item[j]
                if item not in recommended_slate:
                    recommended_slate.append(item)
                    break
                
        return recommended_slate
    
    