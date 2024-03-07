from scipy.io import loadmat, savemat
import numpy as np
import os
import sys
masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo')) # To load all the relevant files
import transforms

def load_lyh_data(sessionId):
    '''
    parse the lyh dataset original data

    Parameters
    ----------
    - sessionId: str: exlusively ("v2", "v3"), which denotes the proper one that will be loaded.
    '''
    # X_tot = []
    # y_tot = []

    dir_path = "data/lyh/originalData"
    left_v2_fp = "left_processed_v2(300).npy"
    left_v3_fp = "left_processed_v3(500).npy"
    right_v2_fp = "right_processed_v2(300).npy"
    right_v3_fp = "right_processed_v3(500).npy"
    leg_v2_fp = "leg_processed_v2(300).npy"
    leg_v3_fp = "leg_processed_v3(500).npy"
    nothing_v2_fp = "nothing_processed_v2(300).npy"
    nothing_v3_fp = "nothing_processed_v3(500).npy"
    
    # get all the data first
    left_v2 = np.load(os.path.join(dir_path, left_v2_fp))
    left_v3 = np.load(os.path.join(dir_path, left_v3_fp))
    right_v2 = np.load(os.path.join(dir_path, right_v2_fp))
    right_v3 = np.load(os.path.join(dir_path, right_v3_fp))
    leg_v2 = np.load(os.path.join(dir_path, leg_v2_fp))
    leg_v3 = np.load(os.path.join(dir_path, leg_v3_fp))
    nothing_v2 = np.load(os.path.join(dir_path, nothing_v2_fp))
    nothing_v3 = np.load(os.path.join(dir_path, nothing_v3_fp))
    eeg_raw_v2 = [left_v2, right_v2, leg_v2]
    eeg_raw_v3 = [left_v3, right_v3, leg_v3]

    fs = 250
    # X_train_tot = []
    # X_test_tot = []
    # y_train_tot = []
    # y_test_tot = []
    X_tot = []
    y_tot = []
    train_ratio = 0.8 # so that 80% trainset, the rest testset
    eeg_raw = eeg_raw_v2 if sessionId == "v2" else eeg_raw_v3

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
        X_raw = tmp[:, :14, :] # filter the channels, only need the first 14 channels
        # (n_trials, 14, 250)
        y_raw = np.array([i for j in range(n_trial)]) # (n_trials,) value = label
        # now shuffle the 300 samples 
        shuffle_idx = np.random.permutation(len(X_raw))
        X_raw = X_raw[shuffle_idx, :, :]
        y_raw = y_raw[shuffle_idx] # although no changes will be made
        X_tot.append(X_raw)
        y_tot.append(y_raw)
        # split_idx = int(n_trial * train_ratio)
        # X_train_tot.append(X_raw[:split_idx])
        # X_test_tot.append(X_raw[split_idx:])
        # y_train_tot.append(y_raw[:split_idx])
        # y_test_tot.append(y_raw[split_idx:])


    X_tot = np.concatenate(X_tot)
    y_tot = np.concatenate(y_tot)
    # X_train_tot = np.concatenate(X_train_tot) # (960, 14, 1000)
    # X_test_tot = np.concatenate(X_test_tot) # (240, 14, 1000)
    # y_train_tot = np.concatenate(y_train_tot) # (960,)
    # y_test_tot = np.concatenate(y_test_tot) # (240,)

    # train_data = X_tot # (1200, 14, 1000)
    # train_label = y_tot.reshape(1200, 1) # (1200, 1)
        
    # allData = train_data # (1200, 14, 1000)
    # allLabel = train_label.squeeze() # (1200, )

    shuffle_num = np.random.permutation(len(X_tot))
    # X_train = X_train_tot[shuffle_num, :, :]
    X = X_tot[shuffle_num, :, :]
    y = y_tot[shuffle_num]
    # y_train = y_train_tot[shuffle_num]
    # shuffle_num = np.random.permutation(len(X_test_tot))
    # X_test = X_test_tot[shuffle_num, :, :]
    # y_test = y_test_tot[shuffle_num]
    # print(f"Shuffle num {shuffle_num}")
    # allData = allData[shuffle_num, :, :]
    # allLabel = allLabel[shuffle_num]

    # X_train, X_test, y_train, y_test = train_test_split(allData, allLabel, train_size=0.8,
    #                                                             random_state=None, shuffle=False)

    # now transpose the dimension to (n_chans, n_times, n_trial)
    # allData = allData.transpose((1, 2, 0))
    # X_train = X_train.transpose((1, 2, 0))
    # X_test = X_test.transpose((1, 2, 0))
    X = X.transpose((1, 2, 0))
    
    # TODO: here, I just put the channels info to None, needs further configuration
    data = {'x': X, 'y': y, 'c': [i for i in range(14)], 's': fs}
    # train_data = {'x': X_train, 'y': y_train, 'c': [i for i in range(14)], 's': fs}
    # test_data = {'x': X_test, 'y': y_test, 'c': [i for i in range(14)], 's': fs}
    #(n_chan, 1000, n_trials)
    return data


def transform_data(X, y):
    """
    Attributes
    ----------
    - X: np.ndarray
        should be in the form of (n_chans, n_times, n_trials) 
    - y: np.ndarray
        should be in the form of (n_trials,)

    Return
    ------
    - X: np.array
        should be in the form of (n_trials, n_chans, n_times, )
    """
    # define transform 
    config = {}
    config['transformArguments'] = {'filterBank':{'filtBank':[[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]],'fs':250, 'filtType':'filter'}}
    transform = transforms.__dict__[list(config['transformArguments'].keys())[0]](**config['transformArguments'][list(config['transformArguments'].keys())[0]])

    # transpose 

    X = X.transpose((2, 0, 1))
    for idx in range(len(X)):
        x = X[idx] 
        print(x.shape)
        dct = {
            'data': x,
            'idx': 114514,
            'label': 123456,
        }
        dct = transform(dct)
        print(dct['data'].shape)
    


if __name__ == "__main__":
    data = load_lyh_data("v3")
    X_test = data['x']
    y_test = data['y']
    print(X_test.shape, y_test.shape)
    transform_data(X_test, y_test)