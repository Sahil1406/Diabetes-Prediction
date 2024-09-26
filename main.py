# Importing All the Libraries


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

# Importing the Data and making it useful

diabetes_data = pd.read_csv('diabetes.csv')

# first 5 rows of the dataset
# print(diabetes_data.head())

# number of rows and columns of the dataset
# print(diabetes_data.shape)


# various statistical values of the data
# print(diabetes_data.describe())

# print(diabetes_data['Outcome'].value_counts())

# 1 is Diabetic Patient
# 0 is Non-Diabetic Patient

# print(diabetes_data.groupby('Outcome').mean())

# Splitting the data from the tag

X = diabetes_data.drop(columns='Outcome' , axis=1)

Y = diabetes_data['Outcome']

# print(X)
# print(Y)


# Data Standardization


scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
# print(standardized_data)
X = standardized_data

# train test split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, stratify= Y, random_state = 2)

# print(X.shape,X_train.shape, X_test.shape)

# training the model

classifier = svm.SVC(kernel = 'linear')

classifier.fit(X_train, Y_train)


# model evaluation

# accuracy score on the training data

x_train_prediction = classifier.predict(X_train)

training_data_accuracy = accuracy_score(x_train_prediction, Y_train)

# print('Accuracy score on training data: ', training_data_accuracy)


# accuracy score on the test data

x_test_prediction = classifier.predict(X_test)

test_data_accuracy = accuracy_score(x_test_prediction, Y_test)

print('Accuracy score on test data: ', test_data_accuracy)


# now if the values needs to be taken from the user

# making a predictive system


input_data1 = (5,166,72,19,175,25.8,0.587,51)

input_data_2 = np.asarray(input_data1)

# reshaping the array as we are predicting one instance

input_data_3 = input_data_2.reshape(1,-1)

# standardizing the input data

input_data = scalar.transform(input_data_3)

# print(input_data)

prediction = classifier.predict(input_data)
# print(prediction)

if (prediction[0] == 1):
    print('The patient is diabetic')
else:
    print('The patient is non-diabetic')