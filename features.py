import numpy as np
import pandas as pd


def logical_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog'] / \
        df['rooms']  # tamhog - size of the household
    # r4t3 - Total persons in the household
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog']  # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms']
    df['v2a1_to_r4t3'] = df['v2a1']/df['r4t3']  # rent to people in household
    df['v2a1_to_r4t3'] = df['v2a1'] / \
        (df['r4t3'] - df['r4t1'])  # rent to people under age 12
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms']  # rooms per person
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize']  # rent to household size
    return df


def engineered_features(df):
    # engineer "normal" values if features are represented by one hot
    col_list = df.columns.tolist()
    for col in col_list:
        partials = ['pared', 'piso', 'techo', 'sanitario', 'energcocinar', 'elimbas',
                    'estadocivil', 'parentesco', 'instlevel', 'tipovivi', 'lugar']
        for s in partials:
            if col.find(s) > -1:
                # TODO: A shit ton of automated feature engineering
                pass
    return df
