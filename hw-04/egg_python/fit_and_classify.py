from numpy import ones
from sklearn import svm
from sklearn.metrics.pairwise import *

def fit_and_classify(train_features, train_labels, test_features):
    clf = svm.SVC(kernel=chi2_kernel, decision_function_shape='ovo').fit(train_features, train_labels)
    return clf.predict(test_features)
