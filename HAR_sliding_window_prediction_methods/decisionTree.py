## Author: Hariharan Seshadri ##
## This script tries to distinguish between SITTING,STANDING, and LAYING labels using Decision Trees ##

import numpy as np
from sklearn import tree
import common


print("Parsing")
#UCI dataset1
# X_train = common.parseFile( 'X_train.txt')
# Y_train = common.parseFile( 'y_train.txt')
#UCI dataset2
X_train = common.parseFile( 'hapt/X_train.txt')
Y_train = common.parseFile( 'hapt/y_train.txt')
#wisdm dataset
# X_train = common.parseCSVFile('X_train.csv')
# Y_train = common.parseCSVFile('y_train.csv')
Y_train = Y_train.flatten()
X_train,Y_train = common.getDataSubset(X_train, Y_train, [1,2,3,4,5,6])

# X_test = common.parseFile('X_test.txt')
# Y_test = common.parseFile('y_test.txt')
X_test = common.parseFile('hapt/X_test.txt')
Y_test = common.parseFile('hapt/y_test.txt')
# X_test = common.parseCSVFile('X_test.csv')
# Y_test = common.parseCSVFile('y_test.csv')
print(X_test.shape)
print(Y_test.shape)
Y_test= Y_test.flatten()
print(Y_test.shape)
X_test,Y_test = common.getDataSubset(X_test, Y_test, [1,2,3,4,5,6])

print("Done")

print("Fitting Data")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train) 

print("Done")
print("Predicting")
accuracy,precision,recall,fscore,fw = common.checkAccuracy(clf.predict(X_train),Y_train,[1,2,3,4,5,6])
print(accuracy)
print(fscore)
print(fw)
print(common.createConfusionMatrix(clf.predict(X_train).flatten(),Y_train,[1,2,3,4,5,6]))
accuracy,precision,recall,fscore,fw = common.checkAccuracy(clf.predict(X_test),Y_test,[1,2,3,4,5,6])
print(common.createConfusionMatrix(clf.predict(X_test).flatten(),Y_test,[1,2,3,4,5,6]))
print(accuracy)
print(fscore)
print(fw)



