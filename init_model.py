import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


def main():
    test, train = data_prep()


if __name__ == '__main__':
    main()
