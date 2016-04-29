from numpy import ones
from sklearn import svm
from sklearn.metrics.pairwise import *
'''
def my_kernel(X, Y):
    """
    We create a custom kernel:

                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)
''' 
    
def fit_and_classify(train_features, train_labels, test_features, k=chi2_kernel):
    clf = svm.SVC(kernel=k, gamma=.03).fit(train_features, train_labels)
    return clf.predict(test_features)
