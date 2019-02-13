import pandas as pd
import numpy as np
import os


def difference(df1, df2):

    # concat dataframes
    df = pd.concat([df1, df2])

    # reset the index
    df = df.reset_index(drop=True)
    # group by
    df_gpby = df.groupby(list(df.columns))

    # reindex
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]

    return df.reindex(idx)


def build_dataset(file, min_rating=0):

    # Read File
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "csvfiles", file)
    df = pd.read_csv(path, delimiter=";",
                     error_bad_lines=False,
                     encoding='ISO-8859-1',
                     low_memory=False
                     )
    # Keep only data with rating higher than min_rating
    df = df[~(df['Rating'] < min_rating)]
    # Sort and keep isbn with 2 or more occurances

    msk = np.random.rand(len(df)) < 0.8
    df_train = df[msk]
    df_test = df[~msk]

    # Brute Force :(
    for x in range(0, 5, 1):
        common_isbn = list(set(df_test['ISBN']).intersection(df_train['ISBN']))

        df_train = df_train.loc[df_train['ISBN'].isin(common_isbn)]
        df_test = df_test.loc[df_test['ISBN'].isin(common_isbn)]

        common_id = list(set(df_test['ID']).intersection(df_train['ID']))

        df_train = df_train.loc[df_train['ID'].isin(common_id)]
        df_test = df_test.loc[df_test['ID'].isin(common_id)]

    df_train = df_train.sort_values(by=['ID'])
    df_test = df_test.sort_values(by=['ID'])

    # create COO
    dictionary = {'train': df_train, 'test': df_test}

    return dictionary
