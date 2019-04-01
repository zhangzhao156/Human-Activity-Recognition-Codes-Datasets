#Author: Nitin A Jain

import numpy as np
from sklearn.svm import SVC,LinearSVC
import common


#Parsing Full training dataset
# XFull = common.parseFile('hapt/X_train.txt')
# YFull = common.parseFile('hapt/y_train.txt')
XFull = common.parseCSVFile('X_train.csv')
YFull = common.parseCSVFile('y_train.csv')
#Parsing Full testing dataset
# XFullTest = common.parseFile('hapt/X_test.txt')
# YFullTest = common.parseFile('hapt/y_test.txt')
XFullTest = common.parseCSVFile('X_test.csv')
YFullTest = common.parseCSVFile('y_test.csv')
#################################################################################################################################
    #Getting the dataset associated with Dynamic Activities on training 
X_Dynamic,Y_Dynamic = common.getDataSubset(XFull,YFull.flatten(),[1,2,3,4,5,6])
    #Getting the dataset associated with Dynamic Activities on testing
X_DynamicTest,Y_DynamicTest = common.getDataSubset(XFullTest,YFullTest.flatten(),[1,2,3,4,5,6])

    #Fitting data using LinearSVC classifier
#clf = LinearSVC(multi_class='crammer_singer')
#clf = SVC(kernel = "linear")
clf=SVC(decision_function_shape='ovo')
clf.fit(X_Dynamic, Y_Dynamic.flatten())
accuracy,precision,recall,fscore,fw = common.checkAccuracy(clf.predict(X_Dynamic),Y_Dynamic,[1,2,3,4,5,6])
print(accuracy)
print(fscore)
print(fw)
print(common.createConfusionMatrix(clf.predict(X_Dynamic).flatten(),Y_Dynamic.flatten(),[1,2,3,4,5,6]))
accuracy,precision,recall,fscore,fw = common.checkAccuracy(clf.predict(X_DynamicTest),Y_DynamicTest,[1,2,3,4,5,6])
print(common.createConfusionMatrix(clf.predict(X_DynamicTest).flatten(),Y_DynamicTest.flatten(),[1,2,3,4,5,6]))
print(accuracy)
print(fscore)
print(fw)
