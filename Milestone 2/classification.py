import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import time


def data_analyze(df, show_corr=0):
    print("#" * 50)
    print("Main Info. About The Data:")
    print(df.info())
    print("#" * 50)
    print("\nMissing Data Exploration:")
    print(df.isnull().sum())
    print("#" * 50 + "\n")

    # Get the correlation between the features
    if show_corr:
        corr = df.corr()
        top_feature = corr.index[abs(corr['user_rating'] > 0.1)]
        plt.subplots(figsize=(12, 8))
        top_corr = df[top_feature].corr()
        sns.heatmap(top_corr, annot=True)
        plt.show()


######################## MAIN ########################
# Loading Data
data = pd.read_csv('AppleStore_training.csv')

# Data Analysis before pre-processing
data_analyze(data, 1)

# Data Pre-Processing
data = data.drop(['id', 'track_name', 'currency', 'ver'], axis=1)  # discard unnecessary features
data['user_rating_ver'] = data['user_rating_ver'].fillna(data['user_rating_ver'].median())  # filling null values of 'user_rating_ver'
data.dropna(how='any', inplace=True)  # dropping rows with null values

print('#' * 50)
print(data.info())  # show the features
print(data['prime_genre'].unique())  # get categories of 'prime_genre'
print('#' * 50)
print(data['cont_rating'].unique())  # get categories 'cont_rating'
print('#' * 50)

i = data[data['prime_genre'] == '0'].index  # get the index of the noisy row which has prime_genre = 0
print(data.loc[i])
data = data.drop(i)  # drop this row

print("\n##### Data analysis after cleaning #####")
data_analyze(data)

X = data.iloc[:, :11]  # Features
Y = data['user_rating']  # Label

# Categories Encoding
le = LabelEncoder()
X['cont_rating'] = le.fit_transform(X['cont_rating'])

col_trans = make_column_transformer((OneHotEncoder(), ['prime_genre']), remainder='passthrough')
X = col_trans.fit_transform(X)

# Features Scaling
standard = StandardScaler()
X = standard.fit_transform(X)

# Splitting Data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)

# models calling
print("\n")


