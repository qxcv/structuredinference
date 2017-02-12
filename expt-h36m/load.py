import numpy as np

from generate_seq_gan import load_data

def loadDataset():
    seq_length = 32
    seq_skip = 3
    train_X, val_X, mean, std = load_data(seq_length, seq_skip, val_subj_5=False)
    dim_observations = train_X.shape[2]

    dataset = {}

    dataset['train']      = train_X
    dataset['mask_train'] = np.ones(train_X.shape[:2])

    dataset['valid']      = val_X
    dataset['mask_valid'] = np.ones(val_X.shape[:2])

    dataset['test']      = dataset['valid']
    dataset['mask_test'] = dataset['mask_valid']

    dataset['dim_observations'] = dim_observations
    dataset['data_type']        = 'real'

    dataset['h36m_mean'] = mean
    dataset['h36m_std'] = std

    return dataset
