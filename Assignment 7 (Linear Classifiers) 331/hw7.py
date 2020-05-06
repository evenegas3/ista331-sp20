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
    # print(x.shape, y.shape)
    return x, y

#32%
def get_train_and_test_sets(X, y):
    limit = 60000
    x_train = X[:limit]
    x_test = X[limit:]
    y_train = y[:limit]
    y_test = y[limit:]
    # print(x_train.shape, x_test.shape)
    # print(y_train.shape, y_test.shape)

    randx = np.random.permutation(limit)
    # randy = np.random.permutation(x_test)

    #2/4 work
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
    # pass
    # pm_copy = cm.copy()
    cm = cm.astype('float')
    
    for i in range(len(cm)):
        cm[i] = cm[i] / sum(cm[i])
        
    cm = cm.round(3)
    return cm
    # count_mtrx = pd.DataFrame(data=cm, columns=list(string.ascii_lowercase)[:10], index=list(string.ascii_lowercase)[:10])
    # pm_correct = np.load('probability_matrix_SGD_correct.npy')
    # print(pm_correct)

    # df = pd.DataFrame(columns=count_mtrx.columns, index=count_mtrx.columns).fillna(0)
    # for row in count_mtrx:
    #     for col in count_mtrx:
    #         prob = count_mtrx.loc[row, col] / count_mtrx.loc[row, row]
    #         df.loc[row, col] = round(prob, 3)

    # print('\nmine')
    # print(df)
    # print(df.to_numpy())

# probability_matrix(np.load('confusion_matrix_SGD_correct.npy'))


def plot_probability_matrices(pm1, pm2, pm3):
    pass

def main():
    x, y = get_data()
    X_train, X_test, y_train, y_test = get_train_and_test_sets(x, y)
    
    sgd = train_to_data(X_train, y_train, 'SGD')
    lr = train_to_data(X_train, y_train, 'LogisticRegression')
    svm = train_to_data(X_train, y_train, 'SVM')

    sgd_cm = confusion_matrix(sgd, X_test, y_test)
    lr_cm = confusion_matrix(lr, X_test, y_test)
    svm_cm = confusion_matrix(svm, X_test, y_test)

    for mod in (('Linear SVM:', probability_matrix(sgd_cm)), ('Logistic Regression:', probability_matrix(lr_cm)), ('Polynomial SVM:', probability_matrix(svm_cm))):
        print(*mod, sep = '\n')

if __name__ == "__main__":
    main()




























    # oil_test = oil.drop(train_idx)
    # limit = 60000
    # x_training, x_testing = x[:limit][:], x[limit:]
    # y_training, y_testing = y[:limit], y[limit:]
    # print(x_training.shape, x_testing.shape)
    # print(y_training.shape, y_testing.shape)

    # x_indices = np.random.permutation(x_training)
    # y_indices = np.random.permutation(y_training)
    # return x_training, x_testing, y_training, y_testing


    # X_train_correct = np.load('X_train_correct.npy')
    # X_test_correct = np.load('X_test_correct.npy')
    # y_train_correct = np.load('y_train_correct.npy')
    # y_test_correct = np.load('y_test_correct.npy')
    # return X_train_correct, X_test_correct, y_train_correct, y_test_correct
# main()    