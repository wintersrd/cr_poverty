import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.layers import GRU, LeakyReLU, Dropout, Dense, GlobalMaxPooling1D,\
    Embedding, Input, multiply

from helpers import val_count, no_yes_fixer, fill_mean
from features import logical_features, engineered_features


WORKING_DIR = '/tmp/cr_poverty/'


def data_prep():
    train_data = pd.read_csv(os.path.join(WORKING_DIR, 'train.csv'))
    test_data = pd.read_csv(os.path.join(WORKING_DIR, 'test.csv'))
    # Remove columns >80% sparse
    drop_cols = [x['col'] for x in val_count(
        train_data) if x['has_null'] is True and x['null_ct'] > 0.8 * train_data.count()[0]]
    train_data.drop(columns=drop_cols, inplace=True)
    test_data.drop(columns=drop_cols, inplace=True)
    # Fix missing and inconsistent values
    train_data = no_yes_fixer(train_data)
    test_data = no_yes_fixer(test_data)
    test_data, train_data = fill_mean(test_data, train_data)
    # Add feature enrichment
    train_data = logical_features(train_data)
    test_data = logical_features(test_data)
    train_data = engineered_features(train_data)
    test_data = engineered_features(test_data)
    return test_data, train_data


def sampler(train_data, sample_rate=0.6, heads_only=False):
    # Get a list of all unique households
    households = train_data.loc[train_data.parentesco1 == 1, 'idhogar'].tolist()
    if heads_only:
        train_data = train_data.loc[train_data.parentesco1 == 1,:]
    train_hh, val_hh = train_test_split(households, train_size=sample_rate)
    model_train = train_data.loc[train_data.idhogar.isin(train_hh), :]
    model_val = train_data.loc[train_data.idhogar.isin(val_hh), :]
    # split into labels and data
    model_train_label = model_train['Target']
    model_train.drop(columns='Target', inplace=True)
    model_val_label = model_val['Target']
    model_val.drop(columns='Target', inplace=True)
    return model_train, model_train_label, model_val, model_val_label


def main():
    test, train = data_prep()
    test.drop(columns='idhogar', inplace=True)
    model_train, model_train_label, model_val, model_val_label = sampler(
        train, 0.8)
    model_train.drop(columns='idhogar', inplace=True)
    model_val.drop(columns='idhogar', inplace=True)


if __name__ == '__main__':
    main()
