
# IN THIS SCRIPT I USED 4 DIFFERENT CLASSIFIER MODELS ALONG WITH DECISION TREE CLASSIFIER
# TO TRAIN A KNOWN DATA SET OF 'HEIGHT', 'WEIGHT' AND 'SHOE SIZE' OF TWO GENDERS
# THEN TEST AND PREDICT WHETHER IF CERTAIN MEASUREMENTS BELONG TO A MALE OR A FEMALE
# THEN I USE ACCURACY METRICS TO DETERMINE WHICH OF THE 5 CLASSIFIER MODELS HAVE THE BEST ACCURACY SCORE
# ALL USING SCIKIT-LEARN PACKAGES

from sklearn import tree #-decision tree classifier
from sklearn.svm import SVC #-Support Vector Classification (SVC classifier)
from sklearn.linear_model import Perceptron #-perception classifier
from sklearn.neighbors import KNeighborsClassifier #-KNeighbors classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score #-to determine accuracy
import numpy as np

# Data set (X) and gender labels (Y)
X = [[185, 82, 45], [166, 62, 37], [175, 72, 41], [155, 53, 38], [165, 63, 41], [192, 92, 45], [177, 65, 33],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['MALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE', 'MALE', 'FEMALE', 'FEMALE', 'FEMALE', 'MALE', 'MALE']

# CALLING CLASSIFIERS
class_tree = tree.DecisionTreeClassifier()
class_SVC = SVC()
class_perc = Perceptron()
class_KNN = KNeighborsClassifier()
class_GSNB = GaussianNB()


# TRAINING THE MODELS USING X, Y DATA SET
class_tree = class_tree.fit(X, Y)
class_SVC = class_SVC.fit(X, Y)
class_perc = class_perc.fit(X, Y)
class_KNN = class_KNN.fit(X, Y)
class_GSNB = class_GSNB.fit(X, Y)

# TESTING PREDICTION USING SAME DATA SET
predict_tree = class_tree.predict(X)
predict_SVC = class_SVC.predict(X)
predict_perc = class_perc.predict(X)
predict_KNN = class_KNN.predict(X)
predict_GSNB = class_GSNB.predict(X)

# TESTING ACCURACY OF THOSE PREDICTION
accu_tree = accuracy_score(Y, predict_tree) * 100 # times 100 to make percentage
print('Decision Tree Accuracy: {}'.format(accu_tree))

accu_SVC = accuracy_score(Y, predict_SVC) * 100
print('SVC Accuracy: {}'.format(accu_SVC))

accu_perc = accuracy_score(Y, predict_perc) * 100
print('Perceptron Accuracy: {}'.format(accu_perc))

accu_KNN = accuracy_score(Y, predict_KNN) * 100
print('KNeighbors accuracy: {}'.format(accu_KNN))

accu_GSNB = accuracy_score(Y, predict_GSNB) * 100
print('GaussianNB accuracy: {}'.format(accu_GSNB))

# CHOOSING THE MOST ACCURATE MODEL
index = np.argmax([accu_tree, accu_SVC, accu_perc, accu_KNN, accu_GSNB])
clf = {0: 'Decision tree', 1: 'SVC', 2: 'Perceptron', 3: 'KNeighbors', 4: 'GaussianNB'}
print('Best gender classifier model is: {}'.format(clf[index]))




