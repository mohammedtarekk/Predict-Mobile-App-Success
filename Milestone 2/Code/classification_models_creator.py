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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle


def data_analyze(df, show_corr=0):
    print("#" * 50)
    print("Main Info. About The Data:")
    print(df.info())
    print("#" * 50)
    print("\nMissing Data Exploration:")
    print(df.isnull().sum())
    print("#" * 50 + "\n")


######################## MAIN ########################
# Loading Data
data = pd.read_csv('AppleStore_training_classification.csv')


# Data Pre-Processing
data = data.drop(['id', 'track_name', 'currency', 'ver'], axis=1)  # discard unnecessary features
data.dropna(axis=0, how="any", thresh=5, inplace=True)  # drop rows with more than 5 null values
data.fillna(data.median(), inplace=True)  # fill null values with its column's median

# tgroba
i = data[data['prime_genre'] == '0'].index  # get the index of the noisy row which has prime_genre = 0
data.loc[i, 'prime_genre'] = None

data["cont_rating"].fillna(method='ffill', inplace=True)
data["prime_genre"].fillna(method='ffill', inplace=True)


# data_analyze(data, 1)
# print(data['prime_genre'].unique())
# print(data['cont_rating'].unique())

X = data.iloc[:, :10]  # Features
Y = data['rate']  # Label

# Categories One-Hot Encoding
X = pd.get_dummies(X, drop_first=False)

# Splitting Data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# Features Scaling
standard = StandardScaler()
standard.fit(x_train)
x_train = standard.transform(x_train)
x_test = standard.transform(x_test)


############## models ##############

# Random Forest
randomForest = RandomForestClassifier(random_state=1)
randomForest.fit(x_train, y_train)
y_prediction = randomForest.predict(x_test)
accuracy = np.mean(y_prediction == y_test) * 100
print("The achieved accuracy using Random Forest Classifier is " + str(accuracy))

# save the model
filename = 'randomForest_classifier.sav'
pickle.dump(randomForest, open(filename, 'wb'))

# Adaboost with decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=100)
bdt.fit(x_train, y_train)
y_prediction = bdt.predict(x_test)
accuracy = np.mean(y_prediction == y_test) * 100
print("The achieved accuracy using Adaboost with Decision Tree is " + str(accuracy))

# save the model
filename = 'adaboost_classifier.sav'
pickle.dump(bdt, open(filename, 'wb'))

# svm
C = 10
rbf_svc = svm.SVC(kernel='rbf', gamma=0.0001, C=C).fit(x_train, y_train)
y_prediction = rbf_svc.predict(x_test)
accuracy = np.mean(y_prediction == y_test) * 100
print("The achieved accuracy using SVM rbf_svc is " + str(accuracy))

# save the model
filename = 'svm_classifier.sav'
pickle.dump(rbf_svc, open(filename, 'wb'))


"""
# KNN
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(x_train, y_train)
y_prediction = knn.predict(x_test)
accuracy = np.mean(y_prediction == y_test) * 100
print("The achieved accuracy using Random Forest Classifier is " + str(accuracy))

# decision tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(x_train, y_train)
y_prediction = clf.predict(x_test)
accuracy = np.mean(y_prediction == y_test) * 100
print("The achieved accuracy using Decision Tree is " + str(accuracy))

# svm
C = 10
svc = svm.SVC(kernel='linear', C=C, gamma=0.01).fit(x_train, y_train)  # minimize hinge oss, One vs One
y_prediction = svc.predict(x_test)
accuracy = np.mean(y_prediction == y_test) * 100
print("The achieved accuracy using SVM svc is " + str(accuracy))

lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train)  # minimize squared hinge loss, One vs All
y_prediction = lin_svc.predict(x_test)
accuracy = np.mean(y_prediction == y_test) * 100
print("The achieved accuracy using SVM lin_svc is " + str(accuracy))

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
"""
