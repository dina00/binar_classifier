import pandas as pd
import numpy as np
from pandas import read_csv
from numpy import nan
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# load the dataset
training = read_csv('training.csv', sep =";", decimal=',')
validation = read_csv('validation.csv', sep =";", decimal=',') # treat comma as decimal point

# plot the no of yes/no
# sns.countplot(validation['classLabel'],label="Label")
# plt.show()

def cleaning(dataframe):
    cols_to_drop = ['variable18'] 
    dataframe.drop(cols_to_drop, inplace=True, axis=1) # drop the column with a huge number of NANs
    dataframe.dropna(axis=0, inplace=True) # drop rows containing NANs
    return dataframe

training = cleaning(training)
validation = cleaning(validation)
print(validation.describe())
print(validation.isnull().sum())

#label encoding; performed betther than hot encoding
def encoding(dataframe):
    cols = ['variable1', 'variable4', 'variable5', 'variable6', 'variable7', 'variable9', 'variable10', 'variable12', 'variable13']
    for i in cols:
        labelencoder = LabelEncoder() # creating instance of labelencoder
        dataframe[i+'_cat'] = labelencoder.fit_transform(dataframe[i])# Assigning numerical values and storing in another column
        dataframe = dataframe.drop([i], axis=1)
    print(dataframe.head(5))
    return dataframe
training['classLabel'] = training['classLabel'].map({'yes.': 1, 'no.': 0})
validation['classLabel'] = validation['classLabel'].map({'yes.': 1, 'no.': 0})
training=encoding(training)
validation=encoding(validation)

print(training.head(5))
print(validation.head(5))

# splitting
y_train= training['classLabel']
y_test= validation['classLabel']

# Create X
X = training.drop(['classLabel'], axis=1)
X_train=X
X= validation.drop(['classLabel'], axis=1)
X_test=X
# check the no of columns and rows
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(y_train.head(20))
print(y_test.head(100))

# ML Alogorithms
# use LogisticRegression
LR = LogisticRegression()
# training using 'training data'
LR.fit(X_train, y_train) # fit the model for training data
prediction_training_lr = LR.predict(X_train)
self_accuracy_lr = accuracy_score(y_train, prediction_training_lr)
print("Logistic Regression Accuracy for training data (self accuracy):", self_accuracy_lr)

# predict the 'target' for 'test data'
prediction_test_lr = LR.predict(X_test)
test_accuracy_lr = accuracy_score(y_test, prediction_test_lr)
print("Logistic Regression Accuracy for test data:", test_accuracy_lr)


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train) # fit the model for training data
prediction_training_dt = decision_tree.predict(X_train)
self_accuracy_dt = accuracy_score(y_train, prediction_training_dt)
print("\nDecision Tree Accuracy for training data (self accuracy):", self_accuracy_dt)

# predict the 'target' for 'test data'
prediction_test_dt = decision_tree.predict(X_test)
test_accuracy_dt = accuracy_score(y_test, prediction_test_dt)
print("Decision Tree Accuracy for test data:", test_accuracy_dt)


SVM = SVC(probability = True)
SVM.fit(X_train, y_train) # fit the model for training data
prediction_training_svm = SVM.predict(X_train)
self_accuracy_svm = accuracy_score(y_train, prediction_training_svm)
print("\nSVM Accuracy for training data (self accuracy):", self_accuracy_svm)

# predict the 'target' for 'test data'
prediction_test_svm = SVM.predict(X_test)
test_accuracy_svm = accuracy_score(y_test, prediction_test_svm)
print("SVM Accuracy for test data:", test_accuracy_svm)


random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train) # fit the model for training data
prediction_training_rf = random_forest.predict(X_train)
self_accuracy_rf = accuracy_score(y_train, prediction_training_rf)
print("\nRandom Forest Accuracy for training data (self accuracy):", self_accuracy_rf)

# predict the 'target' for 'test data'
prediction_test_rf = random_forest.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, prediction_test_rf)
print("Random Forest Accuracy for test data:", test_accuracy_rf)


KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train) # fit the model for training data

# Saving model to disk
pickle.dump(KNN, open('model.pkl','wb'))

prediction_training_knn = KNN.predict(X_train)
self_accuracy_knn = accuracy_score(y_train, prediction_training_knn)
print("\nKNN Accuracy for training data (self accuracy):", self_accuracy_knn)

# predict the 'target' for 'test data'
prediction_test_knn = KNN.predict(X_test)
test_accuracy_knn = accuracy_score(y_test, prediction_test_knn)
print("KNN Accuracy for test data:", test_accuracy_knn)
