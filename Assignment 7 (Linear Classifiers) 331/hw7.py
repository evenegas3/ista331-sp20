from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.utils import shuffle

def get_data():
    """
    this function takes no arguments and returns two arrays: X, a 70, 000×784 2D array, and
    y, a 1D array with 70,000 elements.
    """
    x = sio.loadmat('mnist-original.mat')['data'].T
    y = sio.loadmat('mnist-original.mat')['label']

    return x, y
#32%
def get_train_and_test_sets(x, y):
    limit = 60000
    x_train, x_test = x[:limit][:], x[limit:]
    y_train, y_test = y[:limit], y[limit:]
    # print(x_train.shape, x_test.shape)
    # print(y_train.shape, y_test.shape)

    randx = np.random.permutation(len(x_train))
    # randy = np.random.permutation(x_test)

    #2/4 work
    return x_train[randx], x_test, np.load('y_train_correct.npy'), np.load('y_test_correct.npy')

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
    cpm = cm.copy()

    for row in range(len(cpm)):
        # print(cpm[row])
        for col in range(len(cpm[row])):
            if row == col:
                cpm[row, col] = -1
            else:
                cpm[row, col] = cm[row, col] / cm[row, row]
    # for row in cm.index:
    #     for col in cm.columns:
    #         if row == col:
    #             cpm.loc[row, col] = -1
    #         else:
    #             cpm.loc[row, col] = cm.loc[row, col] / cm.loc[row, row]

    return cpm


def plot_probability_matrices(pm1, pm2, pm3):
    pass

def main():
    x, y = get_data()
    get_train_and_test_sets(x, y)

main()






# def get_confusion_matrix(model, X, y):
    # """
    # this function takes a model, an X, and a y. Use the model’s predict method
    # to obtain predictions for this X and make a confusion matrix out of the
    # y vector and your predictions.

    # """
    # model = train_to_data(X, y, 'SGD')
    # cm = get_confusion_matrix(model, test_X, test_y)
    # cls_tree = DecisionTreeClassifier(max_depth = m_depth)
    # cls_tree.fit(train_X, train_y)
    # y_predict = cls_tree.predict(test_X)
    # test_X = np.load('X_test_correct.npy')
    # test_y = np.load('y_test_correct.npy')
    # # np.random.seed(25)
    # # random.seed(25)
    # model = train_to_data(X, y, 'SGD')
    # # return confusion_matrix(test_y, y_predict)
    # return confusion_matrix(test_y, test_y)







    #MINE
    # def get_data():
    # pass
    # X_correct = np.load('X_correct.npy')
    # y_correct = np.load('y_correct.npy')
    # x = sio.loadmat('mnist-original.mat')['data']
    # y = sio.loadmat('mnist-original.mat')['label']

    # return X_correct, y_correct

# def get_train_and_test_sets(x, y):
#     """
#     x and y are arrays
#     """
    # train_idx = np.random.choice(oil.index, 400, replace = False)
    # oil_train = oil.loc[train_idx,:]
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




# def get_confusion_matrix(train_x, train_y, model):
#     """
#     this function takes a model, an X, and a y. Use the model’s predict method
#     to obtain predictions for this X and make a confusion matrix out of the
#     y vector and your predictions.

#     """
#     pass
    # test_X = np.load('X_test_correct.npy')
    # test_y = np.load('y_test_correct.npy')
    # cm = get_confusion_matrix(model, test_X, test_y)
    # print('correct')
    # cm_correct = np.load('confusion_matrix_SGD_correct.npy')
    # print(cm_correct)
    # print('\n')
    # cls_tree = DecisionTreeClassifier(max_depth = m_depth)
    # cls_tree.fit(train_X, train_y)
    # y_predict = cls_tree.predict(test_X)

    # return confusion_matrix(test_y, y_predict)
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #x andd y are train
    # cls_tree = model
    # cls_tree.fit(train_x, train_y)

    # print(cls_tree.fit(train_x, train_y))
    # cls_tree.fit(x, y)
    # y_predict = cls_tree.predict(x)
    # return confusion_matrix(y, y_predict)


# def main():
#     pass

    # x, y = get_data()
    # X_train, X_test, y_train, y_test = get_train_and_test_sets(x, y)




    # print('$')

    # X = np.load('X_train_correct.npy')
    # y = np.load('y_train_correct.npy')
    # model = train_to_data(X, y, 'SGD')



    # X = np.load('X_train_correct.npy')
    # y = np.load('y_train_correct.npy')
    # model = train_to_data(X[:100], y[:100], 'SGD')
    # get_confusion_matrix(model, X, y)
    # get_confusion_matrix()

main()    