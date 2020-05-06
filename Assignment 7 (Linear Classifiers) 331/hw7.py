from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def get_data():
    X_correct = np.load('X_correct.npy')
    y_correct = np.load('y_correct.npy')
    # x = sio.loadmat('mnist-original.mat')['data']
    # y = sio.loadmat('mnist-original.mat')['label']

    return X_correct, y_correct

def get_train_and_test_sets(x, y):
    """
    x and y are arrays
    """
    limit = 60000
    x_training = x[:limit][:]
    x_testing = x[limit:]

    y_training = y[:limit]
    y_testing = y[limit:]
    # print(x_training.shape, x_testing.shape)
    # print(y_training.shape, y_testing.shape)


def train_to_data(x,y,z):
    # X[:100], y[:100], 'SGD'
    pass

def main():
    x, y = get_data()
    get_train_and_test_sets(x, y)
    # print('$')

main()    
# if __name__ == "__main__":
#     main()