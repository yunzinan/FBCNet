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
from scipy.io import loadmat
import pickle

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
from eegDataset import eegDataset
from baseModel import baseModel
import networks
import transforms
from saveData import fetchData


def load_lyh_data(sessionId, train_idx=10):
    '''
    parse the lyh dataset original data

    Parameters
    ----------
    - sessionId: str
        exlusively ("v2", "v3", "v4"), which denotes the proper one that will be loaded.
    - train_idx: int 
        for splitting train/test set, defaults to 30

    Return
    ------
    - data:
        - 'data': np.ndarray
            (n_trials, n_chans, n_times)
        - 'label': np.ndarray
            (n_trials,)
    '''
    # X_tot = []
    # y_tot = []

    dir_path = "data/lyh/originalData"
    left_v2_fp = "left_processed_v2(300).npy"
    left_v3_fp = "left_processed_v3(500).npy"
    left_v4_fp = "left_processed_v4(500).npy"
    right_v2_fp = "right_processed_v2(300).npy"
    right_v3_fp = "right_processed_v3(500).npy"
    right_v4_fp = "right_processed_v4(500).npy"
    leg_v2_fp = "leg_processed_v2(300).npy"
    leg_v3_fp = "leg_processed_v3(500).npy"
    leg_v4_fp = "leg_processed_v4(500).npy"
    nothing_v2_fp = "nothing_processed_v2(300).npy"
    nothing_v3_fp = "nothing_processed_v3(500).npy"
    nothing_v4_fp = "nothing_processed_v4(500).npy"

    v5_fp = "v5.pkl"
    v5_train_fp = "v5-train.pkl"
    v5_test_fp = "v5-test.pkl"
    v5_1_train_fp = "v5.1-train.pkl"
    v5_1_test_fp = "v5.1-test.pkl"


    # get all the data first
    left_v2 = np.load(os.path.join(dir_path, left_v2_fp))
    left_v3 = np.load(os.path.join(dir_path, left_v3_fp))
    left_v4 = np.load(os.path.join(dir_path, left_v4_fp))
    right_v2 = np.load(os.path.join(dir_path, right_v2_fp))
    right_v3 = np.load(os.path.join(dir_path, right_v3_fp))
    right_v4 = np.load(os.path.join(dir_path, right_v4_fp))
    leg_v2 = np.load(os.path.join(dir_path, leg_v2_fp))
    leg_v3 = np.load(os.path.join(dir_path, leg_v3_fp))
    leg_v4 = np.load(os.path.join(dir_path, leg_v4_fp))
    nothing_v2 = np.load(os.path.join(dir_path, nothing_v2_fp))
    nothing_v3 = np.load(os.path.join(dir_path, nothing_v3_fp))
    nothing_v4 = np.load(os.path.join(dir_path, nothing_v4_fp))
    eeg_raw_v2 = [left_v2, right_v2, leg_v2]
    eeg_raw_v3 = [left_v3, right_v3, leg_v3]
    eeg_raw_v4 = [left_v4, right_v4, leg_v4]
    with open(os.path.join(dir_path, v5_fp), 'rb') as f:
        eeg_raw_v5 = pickle.load(f)
    with open(os.path.join(dir_path, v5_test_fp), 'rb') as f:
        eeg_raw_test_v5 = pickle.load(f)
    with open(os.path.join(dir_path, v5_train_fp), 'rb') as f:
        eeg_raw_train_v5 = pickle.load(f)

    fs = 250
    X_train_tot = []
    X_test_tot = []
    y_train_tot = []
    y_test_tot = []
    sample_trials = 2000 
    base_idx = 0 # (base_idx, base_idx + sample_trials) for sampling
    # X_tot = []
    # y_tot = []
    # eeg_raw = eeg_raw_v2 if sessionId == "v2" else eeg_raw_v3
    if sessionId == "v2":
        eeg_raw = eeg_raw_v2
    elif sessionId == "v3":
        eeg_raw = eeg_raw_v3
    elif sessionId == "v4":
        eeg_raw = eeg_raw_v4
    else:
        eeg_raw = eeg_raw_v5 
        X_train = eeg_raw['train_session']['X_processed']
        y_train = eeg_raw['train_session']['y']
        X_test = eeg_raw['test_session']['X_processed']
        y_test = eeg_raw['test_session']['y']
        # --------------------------------------
        X_test_new = np.array(eeg_raw_test_v5['X_processed']).squeeze()
        y_test_new = np.array(eeg_raw_test_v5['y_true'])
        X_train_new = np.array(eeg_raw_train_v5['X_processed']).squeeze()
        y_train_new = np.array(eeg_raw_train_v5['y_true'])
        X_train_add = np.concatenate([X_train, X_train_new])
        y_train_add = np.concatenate([y_train, y_train_new])
        X_train_add_2 = np.concatenate([X_test, X_train_new])
        y_train_add_2 = np.concatenate([y_test, y_train_new])

        train_data = {'x': X_train, 'y': y_train, 'c': [i for i in range(14)], 's': fs}
        # train_data = {'x': X_train_new, 'y': y_train_new, 'c': [i for i in range(14)], 's': fs}
        # train_data = {'x': X_train_add, 'y': y_train_add, 'c': [i for i in range(14)], 's': fs}
        # train_data = {'x': X_train_add_2, 'y': y_train_add_2, 'c': [i for i in range(14)], 's': fs}
        test_data = {'x': X_test, 'y': y_test, 'c': [i for i in range(14)], 's': fs}
        # test_data = {'x': X_test_new, 'y': y_test_new, 'c': [i for i in range(14)], 's': fs}
        #(n_chan, 1000, n_trials)
        return train_data, test_data

    for i in range(3):
        # XXX: fixed the bug that you cannot simply reshape the files
        # tmp = eeg_raw[i].reshape(15, 300, -1) # (15, 30_0000) => (15, 300, 1000)
        # goal: (15, 30_0000) => (15, 300, 1000)
        trial_list = []
        n_trial = eeg_raw[0].shape[1] // 250 # number of trials in the npy file
        print(f"load {n_trial} trials in the file.")
        for idx in range(n_trial):
            trial_list.append(eeg_raw[i][:, idx * 250 :(idx + 1) * 250]) # [1000:2000]
        # now we have of a list of len 300, w/ each of shape (15, 1000)
        tmp = np.stack(trial_list) # should give a shape of (300, 15, 1000)
        X_raw = tmp[base_idx:base_idx+sample_trials, :14, :] # filter the channels, only need the first 14 channels
        # (n_trials, 14, 250)
        y_raw = np.array([i for j in range(sample_trials)]) # (n_trials,) value = label
        # now shuffle the 300 samples 
        # shuffle_idx = np.random.permutation(len(X_raw))
        # X_raw = X_raw[shuffle_idx, :, :]
        # y_raw = y_raw[shuffle_idx] # although no changes will be made
        # X_tot.append(X_raw)
        # y_tot.append(y_raw)
        X_train_tot.append(X_raw[:train_idx])
        X_test_tot.append(X_raw[train_idx:])
        y_train_tot.append(y_raw[:train_idx])
        y_test_tot.append(y_raw[train_idx:])
        # X_test_tot.append(X_raw[:base_idx])
        # X_test_tot.append(X_raw[base_idx+train_idx:])
        # X_train_tot.append(X_raw[base_idx:base_idx+train_idx])
        # y_test_tot.append(y_raw[:base_idx])
        # y_test_tot.append(y_raw[base_idx+train_idx:])
        # y_train_tot.append(y_raw[base_idx:base_idx+train_idx])


    # X_tot = np.concatenate(X_tot)
    # y_tot = np.concatenate(y_tot)
    X_train_tot = np.concatenate(X_train_tot) # (960, 14, 1000)
    X_test_tot = np.concatenate(X_test_tot) # (240, 14, 1000)
    y_train_tot = np.concatenate(y_train_tot) # (960,)
    y_test_tot = np.concatenate(y_test_tot) # (240,)

    # train_data = X_tot # (1200, 14, 1000)
    # train_label = y_tot.reshape(1200, 1) # (1200, 1)
        
    # allData = train_data # (1200, 14, 1000)
    # allLabel = train_label.squeeze() # (1200, )

    # shuffle_num = np.random.permutation(len(X_tot))
    # X = X_tot[shuffle_num, :, :]
    # y = y_tot[shuffle_num]
    shuffle_num = np.random.permutation(len(X_train_tot))
    X_train = X_train_tot[shuffle_num, :, :]
    y_train = y_train_tot[shuffle_num]
    shuffle_num = np.random.permutation(len(X_test_tot))
    X_test = X_test_tot[shuffle_num, :, :]
    y_test = y_test_tot[shuffle_num]
    # print(f"Shuffle num {shuffle_num}")
    # allData = allData[shuffle_num, :, :]
    # allLabel = allLabel[shuffle_num]

    # X_train, X_test, y_train, y_test = train_test_split(allData, allLabel, train_size=0.8,
    #                                                             random_state=None, shuffle=False)

    # now transpose the dimension to (n_chans, n_times, n_trial)
    # allData = allData.transpose((1, 2, 0))
    # X_train = X_train.transpose((1, 2, 0))
    # X_test = X_test.transpose((1, 2, 0))
    # X = X.transpose((1, 2, 0))
    
    # TODO: here, I just put the channels info to None, needs further configuration
    # data = {'x': X, 'y': y, 'c': [i for i in range(14)], 's': fs}
    train_data = {'x': X_train, 'y': y_train, 'c': [i for i in range(14)], 's': fs}
    test_data = {'x': X_test, 'y': y_test, 'c': [i for i in range(14)], 's': fs}
    #(n_chan, 1000, n_trials)
    return train_data, test_data
    
def load_bci_data(session="train", train_idx=10):
    """
    Attributes
    ----------
    - session: str
        either "train" or "test"

    Return
    ------
    - data: np.ndarray
        should be in the form of (n_trials, n_chans, n_times)
    - label: np.ndarray
        should be in the form of (n_trials,)
    """
    dir_path = "data/bci42a/rawMat"
    train_fp = "s001.mat"
    test_fp = "se001.mat"
    fs = 250

    train_data = loadmat(os.path.join(dir_path, train_fp))
    test_data = loadmat(os.path.join(dir_path, test_fp))

    # print(train_data['x'].shape, train_data['y'].shape)
        
    if session == "train":
        X = train_data['x'].transpose((2, 0, 1))
        shuffle_num = np.random.permutation(len(X))
        X = X[shuffle_num, :, :]
        y = train_data['y'].reshape(288,) 
        y = y[shuffle_num]
        X_train = X[:train_idx, :, :]
        y_train = y[:train_idx]
        X_test = X[train_idx:, :, :]
        y_test = y[train_idx:]
    else :
        X = test_data['x'].transpose((2, 0, 1))
        shuffle_num = np.random.permutation(len(X))
        X = X[shuffle_num, :, :]
        y = test_data['y'].reshape(288,) 
        y = y[shuffle_num]
        X_train = X[:train_idx, :, :]
        y_train = y[:train_idx]
        X_test = X[train_idx:, :, :]
        y_test = y[train_idx:]
        
    train_data = {'x': X_train, 'y': y_train, 'c': [i for i in range(22)], 's': fs}
    test_data = {'x': X_test, 'y': y_test, 'c': [i for i in range(22)], 's': fs}
    #(n_chan, 1000, n_trials)
    return train_data, test_data

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


    # _, train_data = load_lyh_data("v3", train_idx=0)
    # _, train_data = load_bci_data("train", train_idx=0)

    finetune_data, test_data = load_lyh_data("v5", train_idx=10)
    # v5_train_data, v5_test_data = load_lyh_data("v5", train_idx=10)
    # v3_train_data, v3_test_data = load_lyh_data("v4", train_idx=1000)
    # X_train_data = np.concatenate([v3_train_data['x'], v5_train_data['x']]) 
    # X_test_data = np.concatenate([v5_test_data['x']])
    # y_train_data = np.concatenate([v3_train_data['y'], v5_train_data['y']])
    # y_test_data = np.concatenate([v5_test_data['y']])

    # shuffle_num = np.random.permutation(len(X_train_data))
    # X_train_data = X_train_data[shuffle_num, :, ]
    # y_train_data = y_train_data[shuffle_num]
    # shuffle_num = np.random.permutation(len(X_test_data))
    # X_test_data = X_test_data[shuffle_num, :, :]
    # y_test_data = y_test_data[shuffle_num]

    # tot_train_data = {
    #     'x': X_train_data,
    #     'y': y_train_data,
    #     'c': v3_train_data['c'],
    #     's': v3_train_data['s']
    # }
    # tot_test_data = {
    #     'x': X_test_data, 
    #     'y': y_test_data,
    #     'c': v3_test_data['c'],
    #     's': v3_test_data['s'],
    # }
    # finetune_data = tot_train_data
    # test_data = tot_test_data
    # finetune_data, test_data = load_bci_data(session="test", train_idx=16)
    print(finetune_data['x'].shape, test_data['x'].shape)


    # print("------train set w/o finetune----------")
    # ans_list = []
    # tot_cnt = 0
    # acc_cnt = 0

    # d = train_data
    # batch_size = 30
    # tot_batches = d['x'].shape[0] // batch_size


    # for batch_idx in range(tot_batches):
    #     start_idx = batch_idx * batch_size
    #     end_idx = (batch_idx + 1) * batch_size 
    #     data = d['x'][start_idx:end_idx, :, :]
    #     label = d['y'][start_idx:end_idx]
    #     # print(data.shape)
    #     # print(data.shape)
    #     label_list = fbcnet.inference(data) 
    #     # print(label_list)
    #     tot_cnt += label_list.shape[0]
    #     acc_cnt += (label_list == label).sum().item()


    # print(f"tot: {tot_cnt} correct: {acc_cnt} acc: {acc_cnt / tot_cnt}")

    print("------test set w/o finetune----------")
    ans_list = []
    tot_cnt = 0
    acc_cnt = 0

    d = test_data
    batch_size = 30
    tot_batches = d['x'].shape[0] // batch_size


    for batch_idx in range(tot_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size 
        data = d['x'][start_idx:end_idx, :, :]
        label = d['y'][start_idx:end_idx]
        # print(data.shape)
        # print(data.shape)
        label_list = fbcnet.inference(data) 
        # print(label_list)
        tot_cnt += label_list.shape[0]
        acc_cnt += (label_list == label).sum().item()


    print(f"tot: {tot_cnt} correct: {acc_cnt} acc: {acc_cnt / tot_cnt}")
    

    fbcnet.finetune((finetune_data['x'], finetune_data['y']), train_ratio=0.8, earlyStop=False)

    # print("------train set after finetune----------")
    # ans_list = []
    # tot_cnt = 0
    # acc_cnt = 0

    # d = train_data
    # batch_size = 30
    # tot_batches = d['x'].shape[0] // batch_size


    # for batch_idx in range(tot_batches):
    #     start_idx = batch_idx * batch_size
    #     end_idx = (batch_idx + 1) * batch_size 
    #     data = d['x'][start_idx:end_idx, :, :]
    #     label = d['y'][start_idx:end_idx]
    #     # print(data.shape)
    #     # print(data.shape)
    #     label_list = fbcnet.inference(data) 
    #     # print(label_list)
    #     tot_cnt += label_list.shape[0]
    #     acc_cnt += (label_list == label).sum().item()


    # print(f"tot: {tot_cnt} correct: {acc_cnt} acc: {acc_cnt / tot_cnt}")


    print("------test set after finetune----------")
    ans_list = []
    tot_cnt = 0
    acc_cnt = 0

    d = test_data
    batch_size = 30
    tot_batches = d['x'].shape[0] // batch_size


    for batch_idx in range(tot_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size 
        data = d['x'][start_idx:end_idx, :, :]
        label = d['y'][start_idx:end_idx]
        # print(data.shape)
        # print(data.shape)
        label_list = fbcnet.inference(data) 
        # print(label_list)
        tot_cnt += label_list.shape[0]
        acc_cnt += (label_list == label).sum().item()


    print(f"tot: {tot_cnt} correct: {acc_cnt} acc: {acc_cnt / tot_cnt}")

    