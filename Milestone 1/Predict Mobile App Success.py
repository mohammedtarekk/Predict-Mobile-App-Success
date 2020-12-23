import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(degree,X_train,y_train,X_test,y_test):
    poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)
    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(X_test))
    # evaluation
    #print('Training error', metrics.mean_squared_error(y_test, y_train_predicted))
    #print('Co-efficient of linear regression',poly_model.coef_)
    #print('Intercept of linear regression model',poly_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

def linear_regression(x_train,y_train,x_test,y_test):
    cls = linear_model.LinearRegression()
    cls.fit(x_train, y_train)
    # prediction
    prediction = cls.predict(x_test)
    # evaluation
    #print('Co-efficient of linear regression', cls.coef_)
    #print('Intercept of linear regression model', cls.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))


# loading data
data = pd.read_csv('AppleStore_training.csv')
data = data.iloc[:, :]

# data pre-processing
data = data.drop(['id', 'track_name', 'currency', 'ver'], axis=1)  # discard unnecessary features
data.dropna(how='any', inplace=True)  # dropping rows with null values
X = data.iloc[:, :11]  # Features
Y = data['user_rating']  # Label

# One-Hot Encoder
col_trans = make_column_transformer((OneHotEncoder(), ['prime_genre']), remainder='passthrough')  # prime_genre
X = col_trans.fit_transform(X)

col_trans = make_column_transformer((OneHotEncoder(), [30]), remainder='passthrough')  # cont_rating
X = col_trans.fit_transform(X)

# features scaling
standard = StandardScaler()
X = standard.fit_transform(X)

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)

# get the correlation between the features
corr = data.corr()
top_feature = corr.index[abs(corr['user_rating'] > 0.5)]
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
#plt.show()

# models

linear_regression(x_train,y_train,x_test,y_test)
polynomial_regression(3,x_train,y_train,x_test,y_test)


