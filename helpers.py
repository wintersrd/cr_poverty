import pandas as pd
import numpy as np


def val_count(df):
    res = []
    keys = df.columns
    null_ct = df.isnull().sum().values.tolist()
    for i in range(len(keys)):
        rec = {'col': keys[i],
               'null_ct': null_ct[i],
               }
        if null_ct[i] > 0:
            rec['has_null'] = True
        else:
            rec['has_null'] = False
        res.append(rec)
    return res


def no_yes_fixer(df):
    for col in df.columns.tolist():
        if df.loc[df[col].isin(['yes', 'no']), col].count():
            mean_val = np.mean(
                df.loc[~df[col].isin(['yes', 'no']), col].astype('float'))
            df.loc[df[col] == 'no', col] = 0
            # df.loc[df[col] == 'yes', col] = mean_val
            df.loc[df[col] == 'yes', col] = 1
            df[col] = df[col].astype('float')
    return df


def mean_filler(df, col, mean_val=None):
    if not mean_val:
        mean_val = np.mean(df.loc[:, col])
    df = df.fillna({col: mean_val})
    return df, mean_val


def fill_mean(test_data, train_data):
    cols_to_fill = [x['col']
                    for x in val_count(train_data) if x['has_null'] is True]
    cols_to_fill = list(set(
        cols_to_fill + [x['col'] for x in val_count(test_data) if x['has_null'] is True]))
    for col in cols_to_fill:
        train_data, mean_val = mean_filler(train_data, col)
        test_data, _ = mean_filler(test_data, col, mean_val)
    return test_data, train_data
