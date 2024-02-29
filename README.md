# Preliminaries

- follow the `./Instruction` to setup required environments
- the sugguested python version is python3.7


# Usage

1. goto `./FBCNet` directory
2. create foler `./data/lyh/originalData`
3. copy all lyh dataset(including lyh_dataset_v2 AND lyh_dataset_v3) to the folder(`./data/lyh/originalData/`)
4. run the training and evaluation code `python ./code/classify/ho.py 2 'FBCNet'`


If you want to clean the generated data files in `./data/lyh/`. i.e. files in folders such as `rawMat/`, you may run the shell script `./clean.sh` to automatically remove those files and rerun the training commands.


# Results

all training results are tracked in the folder `./output`. 


# Performance

- in-session:
  - lyh_dataset_v2(4class, 300trials/class): test accuracy: 95.42%
  - lyh_dataset_v3(4class, 500trials/class): test accuracy: 99.75% 
