import os
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def preprocess_vote():

    file_name = './datasets/vote.arff'
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    df = df.applymap(lambda x: x.decode('utf-8'))
    df['democrat'] = df['Class'].replace(dict(republican=0, democrat=1)) # democrat_yes
    df = df.drop('Class', axis=1)
    for c in df.columns.drop('democrat'):
        df[c + '_yes'] = df[c].replace({'n': 0, 'y': 1, '?': np.nan})
        df = df.drop(c, axis=1)

    imputer = KNNImputer(n_neighbors=1)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    df_imputed = df_imputed.sort_values('democrat')

    Y = df_imputed['democrat']
    X = df_imputed.drop('democrat', axis=1)

    return X, Y


def preprocess_adult():
    file_name = './datasets/adult.arff'
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'native-country', 'class']
    df[categorical] = df[categorical].applymap(lambda x: x.decode('utf-8'))

    df["workclass"] = df["workclass"].mask(df["workclass"] == '?', np.NaN)
    df["native-country"] = df["native-country"].mask(df["native-country"] == '?', np.NaN)
    df["occupation"] = df["occupation"].mask(df["occupation"] == '?', np.NaN)

    df.dropna(inplace=True)

    # One Hot Encoder
    transformer_one_hot = make_column_transformer(
        (OneHotEncoder(sparse=False),
         ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']),
        remainder='drop')

    transformed = transformer_one_hot.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=transformer_one_hot.get_feature_names_out())

    # Dummy variables
    transformed_df['sex'] = df['sex'].replace({'Male': 0, 'Female': 1}).values  # female = 1, male = 0
    transformed_df['class'] = df['class'].replace({'<=50K': 0, '>50K': 1}).values  # greater than 50K = 1

    # Numerical variables
    transformer_numerical = make_column_transformer((RobustScaler(unit_variance=True), numerical), remainder='drop')
    transformed_numerical = transformer_numerical.fit_transform(df)
    transformed_numerical = pd.DataFrame(transformed_numerical, columns=transformer_numerical.get_feature_names_out())

    df_merged = pd.concat([transformed_df, transformed_numerical], axis=1, sort=False).sample(n=5000, random_state=1)

    Y = df_merged['class']
    X = df_merged.drop('class', axis=1)
    return X, Y

def preprocess_iris():
    file_name = './datasets/iris.arff'
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    df["class"] = df["class"].str.decode('utf-8')

    # Numerical variables
    transformer_numerical = make_column_transformer((StandardScaler(), ['petalwidth', 'petallength', 'sepalwidth', 'sepallength']), remainder='passthrough')
    transformed_numerical = transformer_numerical.fit_transform(df)
    transformed_numerical = pd.DataFrame(transformed_numerical, columns=transformer_numerical.get_feature_names_out())

    Y = transformed_numerical['remainder__class']
    X = transformed_numerical.drop('remainder__class', axis = 1)
    return X, Y