# Optimizing recommendation slates using conditional variational auto-encoders

# Requirements:
We need pytorch framework and python version >= 3.6

The complete list of packages can be imported using anaconda as:
'''
conda env create -f environment.yml
'''

# Dataset:
Data is present inside the data folder under MovieLens folder

# How to run the file: 
Run the train.py file with the following argument:
'''
python3 train.py  --exp_type <"tain" or "evaluate"> 
                  --num_epochs 30 
                  --restore_file <"last" or "best">
                  --pre_weights <analysisCode>
'''

# General code structure:
data/
experiments/
model/
    net.py
    data_loader.py
train.py
evaluate.py
evaluate.py
utils.py

experiments/output_logs/<analysisCode>: contains the loss chart and gradient flow diagrams
experiments/saved_models/<analysisCode>: contains the model parameters corresponding to last and the best model
model/net.py: specifies the neural network architecture, the loss function and evaluation metrics
model/data_loader.py: specifies how the data should be fed to the network
train.py: contains the main training loop
evaluate.py: contains the main loop for evaluating the model
utils.py: utility functions for handling hyperparams/logging/storing model
(Reference: https://cs230-stanford.github.io/pytorch-getting-started.html)

# Specifics to project directory

## For CVAE model:
CVAE: Contains all the code, experiments and analysis results with weights of the model.
Note: by default experiments uses popular movies and slate size of 5, unless not specified

CVAE_UserEmbedding: Implementatin of separate user embedding layer

CVAE_TimeSlates: Implementation of time slates approach

Experiment codes: (
  <analysisCode>
  Analysis1: No initialization 
  Analysis2: Item embeddings initialization
  Analysis4: All layers initialization
  Analysis4_1: With xavier init
  Analysis4_2: Items-embedding pre-trained weights and training
  Analysis4_3: Items-embedding pre-trained weights and training set to false
  Analysis5: Hidden dim from 128 to 16
  Analysi6: All movies with slate size of 5
  Analysis7: Popular movies with slate size of 10
  Analysis8: All movies with slate size of 10
  Analysis9: Popular movies with slate size of of 3
  Analysis10: Popular movies with slate size of 1
  Analysis12_<1-6>: Annealing factors
  Analysis13: Separate user embedding
  Analysis0: With time slates

## For baselines
NCF: For neural collaborative filtering baseline

## Notebooks:
Contains all the train-test-split approach and pre-processing steps including slate generation.

# Authors
* **Gaurav Kumar** - *Initial work* - contact email: s1890219@ed.ac.uk


