# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
Train = pd.read_csv('SalaryData_Train.csv')
Test = pd.read_csv('SalaryData_Test.csv')

string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    Train[i] = number.fit_transform(Train[i])
    Test[i] = number.fit_transform(Test[i])

colnames = Train.columns
len(colnames[0:13])
X_train = Train[colnames[0:13]]
y_train = Train[colnames[13]]
X_test  = Test[colnames[0:13]]
y_test  = Test[colnames[13]]

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

gnb = GaussianNB()
mnb = MultinomialNB()
pred_gnb = gnb.fit(X_train,y_train).predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209))

pred_mnb = mnb.fit(X_train,y_train).predict(X_test)
confusion_matrix(y_test,pred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))
