import os
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def preprocess_vote():

    file_name = './datasets/vote.arff'
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    df = df.applymap(lambda x: x.decode('utf-8'))
    df['democrat'] = df['Class'].replace(dict(republican=0, democrat=1))
    df = df.drop('Class', axis=1)
    for c in df.columns.drop('democrat'):
        df[c + '_yes'] = df[c].replace({'n': 0, 'y': 1, '?': np.nan})
        df = df.drop(c, axis=1)

    imputer = KNNImputer(n_neighbors=1)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

    return df_imputed