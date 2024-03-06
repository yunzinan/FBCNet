"""
define a class as the wrapper of FBCNet
@author: yunzinan 
"""

import ho as HO 


class FBCNet:
    def __init__(self):
        self.loadModel()

    def loadModel(self):
        """
        Load the pre-trained model state dict. 
        """
        print("loading the pre-trained model.")

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
        print("now the data will be inferenced")




if __name__ == "__main__":
    fbcnet = FBCNet()
    fbcnet.loadModel()
    fbcnet.finetune()
    fbcnet.inference()

