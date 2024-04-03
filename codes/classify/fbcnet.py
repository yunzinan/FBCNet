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
import copy
import matplotlib.pyplot as plt

import os
import sys
masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo')) # To load all the relevant files
import transforms

config = {}

config['randSeed'] = 20190821

config['batchSize'] = 30 


config['modelArguments'] = {'nChan': 14, 'nTime': 250, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass':3, 'doWeightNorm': True}
# the config for bci
# config['modelArguments'] = {'nChan': 22, 'nTime': 1000, 'dropoutP': 0.5,
#                                     'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
#                                     'nClass':4, 'doWeightNorm': True}

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
        # loaded_state_dict = torch.load('models/model_state_dict.pth')
        # self.net.load_state_dict(loaded_state_dict)
        print("pre-trained model loaded.")
        # self.model = baseModel(net=self.net, resultsSavePath=None, batchSize=config['batchSize'], setRng=False)

    def train_valid_split(self, X, y, train_ratio=0.8):
        """
        split the (X, y) pairs into training set and validation set

        Attributes
        ----------
        - X: np.ndarray, e.g. (30, 14, 250)
        - y: np.ndarray, e.g. (30,)
        - train_ratio: how much of the input data will be splitted into training set, default is 0.8

        Return
        ------
        - X_train
        - y_train
        - X_valid
        - y_valid
        """
        X_list = [[], [], []]
        y_list = [[], [], []]

        for i in range(len(X)):
            curX = X[i]
            curY = y[i]
            y_list[curY].append(curY) 
            X_list[curY].append(curX) 

        # XXX: shuffle the list 

        for i in range(3):
            shuffle_num = np.random.permutation(len(X_list[i]))
            X_list[i] = np.array(X_list[i])[shuffle_num]
            y_list[i] = np.array(y_list[i])[shuffle_num]

        X_train_list = []
        X_valid_list = []
        y_train_list = []
        y_valid_list = []
        
        for i in range(3):
            # print(len(X_list[i]), len(y_list[i]))
            n = len(X_list[i])
            split_idx = int(n * train_ratio)
            X_train_list.append(X_list[i][:split_idx])
            y_train_list.append(y_list[i][:split_idx])
            X_valid_list.append(X_list[i][split_idx:])
            y_valid_list.append(y_list[i][split_idx:])

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_valid = np.concatenate(X_valid_list)
        y_valid = np.concatenate(y_valid_list)

        print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
        return X_train, y_train, X_valid, y_valid



    def train(self, X_train, y_train, X_valid, y_valid, n_epochs, batch_size, lr=0.001, shuffle=True, earlyStop=True):
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
        - earlyStop: Boolean
            early stop to avoid overfitting, defaults to True
        Return
        ------
        Nothing, but the model state dict will be updated.
        """

        # for validation and early-stopping
        best_loss = float('inf')
        patience = 300 
        train_loss_list = []
        valid_loss_list = []
        train_acc_list = []
        valid_acc_list = []
        best_model_weights = copy.deepcopy(self.net.state_dict())

        # create the Dataloader
        X_train_tensor = X_train.unsqueeze(1)
        y_train_tensor = torch.from_numpy(y_train).to(torch.long)
        if len(X_valid) != 0:
            X_valid_tensor = X_valid.unsqueeze(1)
            y_valid_tensor = torch.from_numpy(y_valid).to(torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if len(X_valid) != 0:
            valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        for epoch in range(n_epochs):
            
            self.net.train()
            tot_loss = 0.0
            num_samples = 0
            correct = 0
            # for inputs, labels in tqdm(train_loader):
            for inputs, labels in train_loader:

                # training
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                tot_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, dim=1)
                num_samples += inputs.size(0)
                correct += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()
            
            avg_loss = tot_loss / num_samples
            acc = correct / num_samples
            train_loss_list.append(avg_loss)
            train_acc_list.append(acc)

            valid_correct = 0

            # validation
            if len(X_valid) != 0 and len(y_valid) != 0:
                self.net.eval()
                tot_loss = 0.0
                num_samples = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        val_outputs = self.net(inputs)
                        _, predicted = torch.max(val_outputs, dim=1)
                        val_loss = criterion(val_outputs, labels)
                        tot_loss += val_loss.item() * inputs.size(0)
                        valid_correct += (predicted == labels).sum().item()
                        num_samples += inputs.size(0)
                
                avg_loss = tot_loss / num_samples
                valid_acc = valid_correct / num_samples
                valid_loss_list.append(avg_loss)
                valid_acc_list.append(valid_acc)

                if avg_loss < best_loss:
                    best_loss = val_loss
                    best_model_weights = copy.deepcopy(self.net.state_dict())
                    patience = 300
                else:
                    patience -= 1
                    if patience == 0 and earlyStop:
                        print(f"Early Stop. Current Epoch: {epoch+1}")
                        break
            
        if len(X_valid) != 0 and len(y_valid) != 0 and earlyStop:
            self.net.load_state_dict(best_model_weights)

        plt.plot(train_loss_list, label='train loss')
        plt.plot(valid_loss_list, label='valid loss')
        plt.plot(train_acc_list, label='train acc')
        plt.plot(valid_acc_list, label='valid acc')

        plt.legend()

        plt.savefig("loss curve.png")

    def finetune(self, data, train_ratio=0.8, earlyStop=True):
        """
        Finetune the model on the given data.

        Attributes
        ----------
        - data: a tuple of two numpy.ndarrays: X and y
            the finetune dataset. should be formatted as
            ((n_trials, n_chans, n_times), (n_trials,)).
            In particular, ((30, 14, 250), (30,))

        - train_ratio: int
            determines how much of the data will be splitted into training set
            the rest will be validation set. defaults to 0.8

        - earlyStop: Boolean
            whether the model will early stop to avoid overfitting, defaults to True

        Return
        ------
        Nothing. but the model state dict will be updated.
        """ 
        # transform the data
        X, y = data  
        X_train, y_train, X_valid, y_valid = self.train_valid_split(X, y, train_ratio=train_ratio)
        X_train = self.transform(X_train)
        if len(X_valid) != 0:
            X_valid = self.transform(X_valid)
        print("the model will be finetuned.") 
        print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
        self.train(X_train, y_train, X_valid, y_valid, n_epochs=1500, lr=0.001, batch_size=30, earlyStop=earlyStop)

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

