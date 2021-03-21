import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Loading Test Data
data = pd.read_csv('AppleStore_training_classification.csv')

# Test Data Pre-Processing
data = data.drop(['id', 'track_name', 'currency', 'ver'], axis=1)  # discard unnecessary features
data.fillna(data.median(), inplace=True)  # fill null values with its column's median

# noisy record
i = data[data['prime_genre'] == '0'].index  # get the index of the noisy row which has prime_genre = 0 , if exists
data.loc[i, 'prime_genre'] = None

data["cont_rating"].fillna(method='ffill', inplace=True)
data["prime_genre"].fillna(method='ffill', inplace=True)

# data_analyze(data, 1)
# print(data['prime_genre'].unique())
# print(data['cont_rating'].unique())

x_test = data.iloc[:, :10]  # Features
y_test = data['rate']  # Label

# Categories One-Hot Encoding
x_test = pd.get_dummies(x_test, drop_first=False)

# Features Scaling
filename = 'standardScaling.sav'
loaded_model = pickle.load(open(filename, 'rb'))
x_test = loaded_model.transform(x_test)


print("########## Random Forest Classifier ##########")
filename = 'randomForest_classifier.sav'
loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict(x_test)
accuracy = np.mean(prediction == y_test) * 100
print("The achieved accuracy using Random Forest Classifier is " + str(accuracy))
print("##############################################\n")


print("########## Adaboost Classifier ##########")
filename = 'adaboost_classifier.sav'
loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict(x_test)
accuracy = np.mean(prediction == y_test) * 100
print("The achieved accuracy using Adaboost Classifier is " + str(accuracy))
print("#########################################\n")


print("########## SVM with Gaussian Kernel Classifier ##########")
filename = 'svm_classifier.sav'
loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict(x_test)
accuracy = np.mean(prediction == y_test) * 100
print("The achieved accuracy using SVM with Gaussian Kernel Classifier is " + str(accuracy))
print("#########################################################\n")