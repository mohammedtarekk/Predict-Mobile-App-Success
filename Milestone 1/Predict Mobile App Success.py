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

mean_error = {}
def linear_regression(x_train, y_train, x_test, y_test):
    cls = linear_model.LinearRegression()
    cls.fit(x_train, y_train)
    # prediction
    prediction = cls.predict(x_test)
    # evaluation
    # print('Co-efficient of linear regression', cls.coef_)
    # print('Intercept of linear regression model', cls.intercept_)
    print('linear_regression: \n Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
    mean_error['linear_regression'] = metrics.mean_squared_error(np.asarray(y_test), prediction)
    plt.figure(figsize=(12, 7))
    sns.regplot(prediction, y_test, color='teal')
    plt.title('linear_regression model')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Actual Ratings')
    plt.show()

def polynomial_regression(degree, X_train, y_train, X_test, y_test):
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
    # print('Training error', metrics.mean_squared_error(y_test, y_train_predicted))
    # print('Co-efficient of linear regression',poly_model.coef_)
    # print('Intercept of linear regression model',poly_model.intercept_)
    print('polynomial_regression: \n Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
    #mean_error['polynomial_regression'] = metrics.mean_squared_error(np.asarray(y_test), prediction)
    plt.figure(figsize=(12, 7))
    sns.regplot(prediction, y_test, color='teal')
    plt.title('polynomial_regression model')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Actual Ratings')
    plt.show()

def svr(x_train, y_train, x_test, y_test):
    model = svm.SVR()
    model.fit(x_train, y_train)
    # prediction
    prediction = model.predict(x_test)
    # evaluation
    # print('Co-efficient of linear regression', model.coef_)
    # print('Intercept of linear regression model', model.intercept_)
    print('SVR: \n Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
    mean_error['SVR'] = metrics.mean_squared_error(np.asarray(y_test), prediction)
    plt.figure(figsize=(12, 7))
    sns.regplot(prediction, y_test, color='teal')
    plt.title('SVR model')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Actual Ratings')
    plt.show()

def randomforest_regression(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    # prediction
    prediction = model.predict(x_test)
    # evaluation
    # print('Co-efficient of linear regression', model.coef_)
    # print('Intercept of linear regression model', model.intercept_)
    print('randomforest_regression: \n Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
    mean_error['randomforest_regression'] = metrics.mean_squared_error(np.asarray(y_test), prediction)
    plt.figure(figsize=(12, 7))
    sns.regplot(prediction, y_test, color='teal')
    plt.title('randomforest_regression model')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Actual Ratings')
    plt.show()

def ridge(alpha,x_train, y_train, x_test, y_test):
    model = Ridge(alpha=alpha)
    # fit model
    model.fit(x_train, y_train)
    # prediction
    prediction = model.predict(x_test)
    print('Ridge: \n Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
    mean_error['Ridge'] = metrics.mean_squared_error(np.asarray(y_test), prediction)
    plt.figure(figsize=(12, 7))
    sns.regplot(prediction, y_test, color='teal')
    plt.title('Ridge model')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Actual Ratings')
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
linear_regression(x_train, y_train, x_test, y_test)
polynomial_regression(2, x_train, y_train, x_test, y_test)
svr(x_train, y_train, x_test, y_test)
randomforest_regression(x_train, y_train, x_test, y_test)
ridge(0.5, x_train, y_train, x_test, y_test)
mean_error = pd.DataFrame.from_dict(mean_error,orient = 'index')
mean_error.sort_values(by = 0, inplace = True)
mean_error.plot(kind = 'barh')
plt.show()
