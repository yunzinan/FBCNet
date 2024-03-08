"""
define a class as the wrapper of FBCNet
@author: yunzinan 
"""

import ho as HO 
from ho import networks
from ho import baseModel
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import os
import sys
masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo')) # To load all the relevant files
import transforms

config = {}

config['randSeed'] = 20190821

config['batchSize'] = 30 


# config['modelArguments'] = {'nChan': 14, 'nTime': 250, 'dropoutP': 0.5,
#                                     'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
#                                     'nClass':3, 'doWeightNorm': True}
# the config for bci
config['modelArguments'] = {'nChan': 22, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass':4, 'doWeightNorm': True}

# Training related details    
config['modelTrainArguments'] = {'stopCondi':  {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500, 'varName' : 'epoch'}},
          'c2': {'NoDecrease': {'numEpochs' : 200, 'varName': 'valInacc'}} } }},
          'classes': [0,1,2], 'sampler' : 'RandomSampler', 'loadBestModel': True,
          'bestVarToCheck': 'valInacc', 'continueAfterEarlystop':True,'lr': 1e-3}


class FBCNet:
    def __init__(self):
        self.network = networks.__dict__['FBCNet']
        self.device = torch.device("cuda")
        self.setRandom(config['randSeed']) 
        self.loadModel()

    def transform(self, X):
        """
        Transform X using filter bank.
        Attributes
        ----------
        - X: np.ndarray
            should be in the form of (n_trials, n_chans, n_times)
        
        Return
        ------
        - X: torch.Tensor
            should be in the form of (n_trials, n_chans, n_times, 9), where 9 denotes the filter banks.
        """
        config = {}
        config['transformArguments'] = {'filterBank':{'filtBank':[[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]],'fs':250, 'filtType':'filter'}}
        transform = transforms.__dict__[list(config['transformArguments'].keys())[0]](**config['transformArguments'][list(config['transformArguments'].keys())[0]])
        X_list = []
        # print(X)/
        for idx in range(len(X)):
            x = X[idx]
            dct = {
                'data': x,
                'label': 114514, # whatever 
                'idx': 1919810, # whatsoever
            }
            dct = transform(dct)
            X_list.append(dct['data'])
    
        X_transformed = torch.stack(X_list)
        return X_transformed


    def setRandom(self, seed):
        '''
        Set all the random initializations with a given seed

        '''
        # Set np
        np.random.seed(seed)

        # Set torch
        torch.manual_seed(seed)

        # Set cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def loadModel(self):
        """
        Load the pre-trained model state dict. 
        """
        print("loading the pre-trained model.")
        self.net = self.network(**config['modelArguments'])
        print('Trainable Parameters in the network are: ' + str(self.count_parameters()))
        loaded_state_dict = torch.load('models/bci_model_state_dict.pth')
        # self.net.load_state_dict(loaded_state_dict)
        print("pre-trained model loaded.")
        self.model = baseModel(net=self.net, resultsSavePath=None, batchSize=config['batchSize'], setRng=False)

    def train(self, X_train, y_train, n_epochs, batch_size, lr=0.001, shuffle=True):
        """
        train the models for n_epochs

        Attributes
        ----------
        - X_train: torch.Tensor
            should be in the form of (n_trials, n_chans, n_times, 9)
        - y_train: torch.Tensor 
            should be in the form of (n_trials,)
        - n_epochs: int
            the number of total train epochs
        - batch_size: int
        - lr: float
        - shuffle: Boolean
            whether the dataloader will be shuffled or not, defaults to True
        Return
        ------
        Nothing, but the model state dict will be updated.
        """

        # create the Dataloader
        X_tensor = X_train.unsqueeze(1)
        y_tensor = torch.from_numpy(y_train)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        for epoch in range(n_epochs):
            self.net.train()

            # for inputs, labels in tqdm(train_loader):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.net(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()



    def finetune(self, data):
        """
        Finetune the model on the given data.

        Attributes
        ----------
        - data: a tuple of two numpy.ndarrays: X and y
            the finetune dataset. should be formatted as
            ((n_trials, n_chans, n_times), (n_trials,)).
            In particular, ((30, 14, 250), (30,))

        Return
        ------
        Nothing. but the model state dict will be updated.
        """ 
        # transform the data
        X, y = data  
        X = self.transform(X)
        print("the model will be finetuned.") 
        print(X.shape, y.shape)
        self.train(X, y, n_epochs=1500, lr=0.001, batch_size=16)

    def inference(self, data):
        """
        predict labels of the unlabeled data.

        Attributes:
        ----------
        - data: numpy.ndarray
            the unlabeled data for inference. should be formatted
            as (1, n_chans, n_times). In particular, (1, 14, 250)

        Return:
        ------
        - label: int 
            the predicted label for the input data
        """
        # tranform the input
        data = self.transform(data) # should return a Tensor of (1, n_chans, n_times, 9) 
        # print("now the data will be inferenced")
        self.net.eval()
        d = data.unsqueeze(1)
        with torch.no_grad():
            # print(d.shape)
            preds = self.net(d.to(self.device))

            _, preds = torch.max(preds, 1)

        return preds.cpu().numpy()



if __name__ == "__main__":
    fbcnet = FBCNet()
    fbcnet.loadModel()
    fbcnet.finetune()
    fbcnet.inference()

