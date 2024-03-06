"""
define a class as the wrapper of FBCNet
@author: yunzinan 
"""

import ho as HO 
from ho import networks
from ho import baseModel
import numpy as np
import torch

config = {}

config['randSeed'] = 20190821

config['batchSize'] = 30


config['modelArguments'] = {'nChan': 14, 'nTime': 250, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass':3, 'doWeightNorm': True}


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
        loaded_state_dict = torch.load('models/model_state_dict.pth')
        self.net.load_state_dict(loaded_state_dict)
        print("pre-trained model loaded.")
        self.model = baseModel(net=self.net, resultsSavePath=None, batchSize=config['batchSize'])
        self.model

    def finetune(self, data):
        """
        Finetune the model on the given data.

        Attributes:
        ----------
        - data: a tuple of two numpy.ndarrays: X and y
            the finetune dataset. should be formatted as
            ((n_trials, n_chans, n_times), (n_trials,)).
            In particular, ((30, 14, 250), (30,))

        Return:
        ------
        Nothing. but the model state dict will be updated.
        """ 
        print("the model will be finetuned.") 
        # self.model.train()

    def inference(self, data):
        """
        predict labels of the unlabeled data.

        Attributes:
        ----------
        - data: numpy.ndarray
            the unlabeled data for inference. should be formatted
            as (1, n_trials, n_times). In particular, (1, 14, 250)

        Return:
        ------
        - label: int 
            the predicted label for the input data
        """
        # print("now the data will be inferenced")
        d = torch.from_numpy(data).unsqueeze(1)
        with torch.no_grad():
            preds = self.net(d.to(self.device))

            _, preds = torch.max(preds, 1)

        return preds.cpu().numpy()



if __name__ == "__main__":
    fbcnet = FBCNet()
    fbcnet.loadModel()
    fbcnet.finetune()
    fbcnet.inference()

