import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
"""
Erick Venegas
ISTA 331
04-28-20
Sean Current

Collab: Yolanda Terrazas
"""

def get_classification_frames():
    """
    This function takes no arguments. It creates a DataFrame from both the training.csv and testing.csv files.
    Using only the first 10 columns of the dataframes.

    PARAMETERS: N/A

    RETURNS: train -- a df read from 'training.csv'
             test -- a df read from 'testing.csv'
    """
    train = pd.read_csv('training.csv').iloc[:,0:10]
    test = pd.read_csv('testing.csv').iloc[:,0:10]

    return train, test

def get_X_and_y(df):
    """
    This function takes a frame and returns a DataFrame of predictors(x) and a vector (y) of class labels.

    PARAMETERS: df -- a dataframe from get_classification_frames()

    RETURNS: x -- DataFrame of predictors
             y -- Vector (Series) of class labels
    """
    x = df.copy()
    y = x['class']
    del x['class']

    return x, y

def make_and_test_tree(train_X, train_y, test_X, test_y, m_depth):
    """
    This function takes 5 arguments: a training X, a training y, a testing X, a testing y,
    and a maximum depth. Initialize and fit a DecisionTreeClassifier on the training data
    with the given maximum depth. Return a confusion matrix measuring the accuracy of the model on the testing data.

    PARAMETERS: train_X -- a dataset of training x data
                train_y -- a dataset of training y data
                test_y -- a dataset of testing y data
                m_depth -- an integer, max depth used to calc. accuracy

    RETURNS: a confusion matrix of measuring the accuracy of the model on the testing data
    """
    cls_tree = DecisionTreeClassifier(max_depth = m_depth)
    cls_tree.fit(train_X, train_y)
    y_predict = cls_tree.predict(test_X)

    return confusion_matrix(test_y, y_predict)

def plot_confusion_matrix(train_X, train_y, test_X, test_y, m_depth):
    """
    This function takes the same 5 arguments as the previous function.
    Get a confusion matrix from the previous function and display it using plt.matshow.
    Pass the parameter cmap = plt.cm.gray. Don’t call plt.show() in this function; you’ll call it later in main.

    PARAMETERS: train_X -- a dataset consisting of training x data
                train_y -- a dataset consisting of training y data
                test_X -- a dataset consisting of test x data
                m_depth -- an integer, consists of the max depth for 
                accuracy(used as a parameter in make_and_test() function)
    """
    cm = make_and_test_tree(train_X, train_y, test_X, test_y, m_depth)
    plt.matshow(cm, cmap=plt.cm.gray)
    print('$$$$$$$')

def get_regression_frame():
    """
    This function takes no arguments and creates a DataFrame from the bikes.csv file

    PARAMETERS: N/A

    RETURNS: df -- a dataframe from bikes.csv file
    """
    return pd.read_csv('bikes.csv')

def get_regression_X_and_y(data):
    """
    This function takes the frame created by get regression frame and splits it into training
    and testing X and y. Use np.random.choice to select a random sample of 15000
    instances to be the training set. Take the rest to be the testing set.

    PARAMETERS: data -- data, a dataframe made from the 'bikes.csv' file

    PARAMETERS: train_x -- a dataset of training data for x
                test_x -- a dataset of training data for x
                train_y -- a dataset of training data for y
                test_y -- dataset of test data for y

    RETURNS: training X, testing X, training y, testing y in that order.
    """
    lis = [x for x in range(len(data))]
    dropped = ['casual', 'cnt', 'datetime', 'registered']
    choice = np.random.choice(lis, size=15000, replace=False)

    train_x, test_x = data.iloc[choice], data.drop(choice)
    train_y, test_y = train_x['casual'], test_x['casual']
    
    train_x, test_x = train_x.drop(columns=dropped), test_x.drop(columns=dropped)

    return train_x, test_x, train_y, test_y

def make_depth_plot(X, y, n, key):
    """
    this function takes four arguments: a X and a y, a maximum depth n, and a
    keyword representing the model type, either ’tree’ or ’forest’. For each integer i between 1 and n inclusive,
    initalize a model with max depth = i. Make a DecisionTreeRegressor if the keyword is tree, or a
    RandomForestRegressor if the keyword is ’forest’. Use cross val score with cv = 5 (five-fold-cross-validation)
    and scoring = ’neg mean squared error’ to evaluate the model on the training data. 

    PARAMETERS: X -- dataset of x data
                y -- dataset of y data
                n -- an integer, the max_depth for accuracy
                key -- the model type, determine whether you want
    
    RETURNS: The max score at position i
    """
    lis = []
    for i in range(1, n+1):
        if key == 'tree':
            model = DecisionTreeRegressor(max_depth=i)
        if key == 'forest':
            model = RandomForestRegressor(n_estimators=25, max_depth=i)
        score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        lis.append(score)
    
    lis = np.array(lis)
    meanl = lis.mean(axis=1)
    sd = lis.std(axis=1)

    plt.errorbar(np.arange(len(meanl)), meanl, (sd/np.sqrt(5)))

    return meanl.argmax()+1

def compare_regressors(train_X, train_y, test_X, test_y, lis):
    """
    This function takes a training X, a training y, testing X, a testing y, and a list containing a
    DecisionTreeRegressor and a RandomForestRegressor (already fit to the training data).
    For each model, compute its MSE on the training set.

    PARAMETERS: train_X -- a dataset consisting of training_x data
                train_y -- a dataset consisting of training_y data
                test_X -- a dataset consisting of testing_x data
                test_y -- a dataset consisting of testing_y data
                lis -- a list containing DecisionTreeRegressor and RandomForestRegressor

    RETURNS: N/A
    """
    tree_mse = np.sum((lis[0].predict(train_X) - train_y) ** 2) / len(train_y)
    rfr_mse = np.sum((lis[1].predict(train_X) - train_y) ** 2) / len(train_y)

    tree_r2 = 1-tree_mse/np.var(train_y)
    forest_r2 = 1-rfr_mse/np.var(train_y)

    tree_rmse = np.sqrt(np.sum((lis[0].predict(test_X) - test_y) ** 2) / len(test_y))
    rfr_rmse = np.sqrt(np.sum((lis[1].predict(test_X) - test_y) ** 2) / len(test_y))

    print('-----------------------------------')
    print('Model type:  ','DecisionTreeRegressor')
    print('Depth:       ',lis[0].max_depth)
    print('R^2:         ',np.round(tree_r2,4))
    print('Testing RMSE:',np.round(tree_rmse,4))

    print('-----------------------------------')
    print('Model type:  ','RandomForestRegressor')
    print('Depth:       ',lis[1].max_depth)
    print('R^2:         ',np.round(forest_r2, 4))
    print('Testing RMSE:',np.round(rfr_rmse,4))

    return None

def main():
    train, test = get_classification_frames()
    train_x = get_X_and_y(train)[0]
    train_y = get_X_and_y(train)[1]

    test_x = get_X_and_y(test)[0]
    test_y = get_X_and_y(test)[1]

    plot_confusion_matrix(train_x, train_y, test_x, test_y, 1)
    plt.show()
    # plot_confusion_matrix(train_x, train_y, test_x, test_y, 5)
    # plt.show()
    # regression = get_regression_frame()
    # trX, teX, trY, teY = get_regression_X_and_y(regression)
    # x = make_depth_plot(teX, teY, 15, 'tree')
    # plt.show()
    # x2 = make_depth_plot(teX, teY, 15, 'forest')
    # plt.show()
    # dec = DecisionTreeRegressor(max_depth=x)
    # dec.fit(teX, teY)
    # ran = RandomForestRegressor(max_depth=x2)
    # ran.fit(teX, teY)
    # l = []
    # l.append(dec)
    # l.append(ran)
    # compare_regressors(trX, trY, teX, teY, l)


if  __name__ == "__main__":
    main()
    # print('$$$$$')