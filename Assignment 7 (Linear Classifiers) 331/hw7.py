from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.utils import shuffle
import pandas as pd
import string

def get_data():
    """
    this function takes no arguments and returns two arrays: X, a 70, 000×784 2D array, and
    y, a 1D array with 70,000 elements.
    """
    d = sio.loadmat('mnist-original.mat')
    x = d['data'].T
    y = d['label'][0]

    return x, y

def get_train_and_test_sets(X, y):
    limit = 60000
    x_train = X[:limit]
    x_test = X[limit:]
    y_train = y[:limit]
    y_test = y[limit:]
    randx = np.random.permutation(limit)

    return x_train[randx], x_test, y_train[randx], y_test

def train_to_data(train_x, train_y, model_name):
    """
    this function takes a training X, a training y, and a string containing the name
    of the model. If the model name is ’SGD’, make a SGDClassifier with max iter = 200 and a
    tolerance of 0.001 (Remember the default SGDClassifier is a linear SVM). If the model name is
    ’SVM’, make a SVC with kernel = ’poly’. Otherwise, make a LogisticRegression with multi class
    = ’multinomial’ and solver = ’lbfgs’.
    """
    if model_name == 'SGD':
        clas = SGDClassifier(alpha=0.0001, tol=0.001, max_iter=200)
        clas.fit(train_x[:10000], train_y[:10000])
    elif model_name == 'LogisticRegression':
        clas = LogisticRegression(C=1.0, tol=0.0001, max_iter=100, multi_class='multinomial', solver='lbfgs')
        clas.fit(train_x, train_y)
    elif model_name == 'SVM':
        clas = SVC(C=1.0, tol=0.001, kernel='poly')
        clas.fit(train_x[:10000], train_y[:10000])
    return clas

def get_confusion_matrix(model, x, y):
    """
    this function takes a model, an X, and a y. Use the model’s predict method
    to obtain predictions for this X and make a confusion matrix out of the y vector and your predictions.
    Return the matrix
    """
    return confusion_matrix(y, model.predict(x))

def probability_matrix(cm):
    cm = cm.astype('float')

    for i in range(len(cm)):
        cm[i] = cm[i] / sum(cm[i])
    cm = cm.round(3)

    return cm

def plot_probability_matrices(pm1, pm2, pm3):
    fig, (lin, logreg, poly) = plt.subplots(1, 3)

    lin.matshow(pm1, cmap='binary')
    lin.set_title('Linear SVM\n')

    logreg.matshow(pm2, cmap='binary')
    logreg.set_title('Logistic Regression\n')

    poly.matshow(pm3, cmap='binary')
    poly.set_title('Polynomial SVM\n')


def main():
    x, y = get_data()
    X_train, X_test, y_train, y_test = get_train_and_test_sets(x, y)
    
    sgd = train_to_data(X_train, y_train, 'SGD')
    lr = train_to_data(X_train, y_train, 'LogisticRegression')
    svm = train_to_data(X_train, y_train, 'SVM')

    sgd_cm = get_confusion_matrix(sgd, x, y)
    lr_cm = get_confusion_matrix(lr, x, y)
    svm_cm = get_confusion_matrix(svm, x, y)

    m1 = probability_matrix(sgd_cm)
    m2 = probability_matrix(lr_cm)
    m3 = probability_matrix(svm_cm)

    plot_probability_matrices(m1, m2, m3)

    for mod in (('Linear SVM:', probability_matrix(sgd_cm)), ('Logistic Regression:', probability_matrix(lr_cm)), ('Polynomial SVM:', probability_matrix(svm_cm))):
        print(*mod, sep = '\n')

    plt.show()

if __name__ == "__main__":
    main()
