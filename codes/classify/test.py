import fbcnet

import numpy as np
import torch
import sys
import os
import time
import xlwt
import csv
import random
import math
import copy
from torch.utils.data import DataLoader

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
from eegDataset import eegDataset
from baseModel import baseModel
import networks
import transforms
from saveData import fetchData




def get_data():
    """
    fetch the lyh dataset 
    """
    config = {}

    toolboxPath = os.path.dirname(masterPath)
    config['inDataPath'] = os.path.join(toolboxPath, 'data')

    modeInFol = 'multiviewPython'

    config['inDataPath'] = os.path.join(config['inDataPath'], 'lyh', modeInFol)

    # Path to the input data labels file
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    fetchData(os.path.dirname(config['inDataPath']), 2) # Make sure that all the required data is present!
    print("Data loading finished")
    data = eegDataset(dataPath = config['inDataPath'], dataLabelsPath= config['inLabelPath'], preloadData = False, transform=None)

        
    trainData = copy.deepcopy(data)
    testData = copy.deepcopy(data)

    if len(data.labels[0])>4:
        idxTrain = [i for i, x in enumerate(data.labels) if x[4] == '0' ]
        idxTest = [i for i, x in enumerate(data.labels) if x[4] == '1' ]
    else:
        raise ValueError("The data can not be divided based on the sessions")

    testData.createPartialDataset(idxTest)
    finetuneData = copy.deepcopy(testData)
    finetuneData.createPartialDataset(list(range(30)))
    testData.createPartialDataset(list(range(30, len(testData))))
    trainData.createPartialDataset(idxTrain)

    

    return trainData, finetuneData, testData

    return data


if __name__ == "__main__":
    fbcnet = fbcnet.FBCNet()

    trainData, finetuneData, testData = get_data()


    # predicted = []
    # actual = []
    # loss = 0
    # batch_size = 80 
    # totalCount = 0
    # # set the network in the eval mode
    print("------train set w/o finetune----------")
    dataLoader = DataLoader(trainData, batch_size=1, shuffle=False)

    ans_list = []
    tot_cnt = 0
    acc_cnt = 0

    for d in dataLoader:
        data = d['data'].numpy()
        label = d['label'].numpy()
        # print(data.shape)
        l = fbcnet.inference(data) 
        tot_cnt += 1
        if l == label:
            acc_cnt += 1
        ans_list.append(l[0])

    print(f"tot: {tot_cnt} correct: {acc_cnt} acc: {acc_cnt / tot_cnt}")

    print("------test set w/o finetune----------")
    dataLoader = DataLoader(testData, batch_size=1, shuffle=False)

    ans_list = []
    tot_cnt = 0
    acc_cnt = 0

    for d in dataLoader:
        data = d['data'].numpy()
        label = d['label'].numpy()
        # print(data.shape)
        l = fbcnet.inference(data) 
        tot_cnt += 1
        if l == label:
            acc_cnt += 1
        ans_list.append(l[0])

    print(f"tot: {tot_cnt} correct: {acc_cnt} acc: {acc_cnt / tot_cnt}")
    
    # finetuning the model
    dataLoader = DataLoader(finetuneData, batch_size=30, shuffle=False)

    for d in dataLoader:
        data = d['data'].numpy()
        label = d['label'].numpy()

        # print(data.shape, label.shape)    
        fbcnet.finetune((data, label))


    print("------test set after finetune----------")
    dataLoader = DataLoader(testData, batch_size=1, shuffle=False)

    ans_list = []
    tot_cnt = 0
    acc_cnt = 0

    for d in dataLoader:
        data = d['data'].numpy()
        label = d['label'].numpy()
        # print(data.shape)
        l = fbcnet.inference(data) 
        tot_cnt += 1
        if l == label:
            acc_cnt += 1
        ans_list.append(l[0])

    print(f"tot: {tot_cnt} correct: {acc_cnt} acc: {acc_cnt / tot_cnt}")

    # fbcnet.inference()
    