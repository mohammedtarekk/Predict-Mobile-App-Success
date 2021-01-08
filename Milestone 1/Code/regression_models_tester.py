import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Loading Test Data
data = pd.read_csv('AppleStore_training.csv')

# Test Data Pre-Processing
data = data.drop(['id', 'track_name', 'currency', 'ver'], axis=1)  # discard unnecessary features
data.fillna(data.median(), inplace=True)

i = data[data['prime_genre'] == '0'].index  # get the index of the noisy row which has prime_genre = 0
data.loc[i, 'prime_genre'] = None

data["cont_rating"].fillna(method='ffill', inplace=True)
data["prime_genre"].fillna(method='ffill', inplace=True)


X_test = data.iloc[:, :11]  # Features
Y_test = data['user_rating']  # Label

#print(data.median())
#print(data['cont_rating'].unique())
#print(data['prime_genre'].unique())

# Categories Encoding
filename = 'cont_rating_labelEncoder.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X_test['cont_rating'] = loaded_model.transform(X_test['cont_rating'])

filename = 'prime_genre_OHEncoder.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X_test = loaded_model.transform(X_test)

# Features Scaling
filename = 'standardScaling.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X_test = loaded_model.transform(X_test)


print("########## Linear Regression ##########")
filename = 'linear_regression.sav'
loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict(X_test)
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
score = loaded_model.score(X_test, Y_test)
print("Test score: {0:.2f} %".format(100 * score))
print("#######################################\n")


print("########## Polynomial Regression ##########")
filename = 'poly_regression.sav'
loaded_model = pickle.load(open(filename, 'rb'))
poly_features = PolynomialFeatures(degree=2)
X_test_poly = poly_features.fit_transform(X_test)

prediction = loaded_model.predict(X_test_poly)
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
score = loaded_model.score(X_test_poly, Y_test)
print("Test score: {0:.2f} %".format(100 * score))
print("##########################################\n")


print("########## SVM Regression ##########")
filename = 'svm_regression.sav'
loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict(X_test)
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
score = loaded_model.score(X_test, Y_test)
print("Test score: {0:.2f} %".format(100 * score))
print("#######################################\n")


print("########## RandomForest Regression ##########")
filename = 'randomforest_regression.sav'
loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict(X_test)
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
score = loaded_model.score(X_test, Y_test)
print("Test score: {0:.2f} %".format(100 * score))
print("#######################################\n")


print("########## Ridge Regression ##########")
filename = 'ridge_regression.sav'
loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict(X_test)
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
score = loaded_model.score(X_test, Y_test)
print("Test score: {0:.2f} %".format(100 * score))
print("#######################################\n")
