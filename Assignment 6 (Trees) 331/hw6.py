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
"""

def get_classification_frames():
    """
    """
    train = pd.read_csv('training.csv').iloc[:,0:10]
    test = pd.read_csv('testing.csv').iloc[:,0:10]

    return train, test

def get_X_and_y(df):
    x = df.copy()
    y = x['class']
    del x['class']

    return x, y

def make_and_test_tree(train_X, train_y, test_X, test_y, m_depth):
    cls_tree = DecisionTreeClassifier(max_depth = m_depth)
    cls_tree.fit(train_X, train_y)
    y_predict = cls_tree.predict(test_X)

    return confusion_matrix(test_y, y_predict)

def plot_confusion_matrix(train_X, train_y, test_X, test_y, m_depth):
    cm = make_and_test_tree(train_X, train_y, test_X, test_y, m_depth)
    plt.matshow(cm, cmap=plt.cm.grey)

def get_regression_frame():
    return pd.read_csv('bikes.csv')

def get_regression_X_and_y(data):
    lis = [x for x in range(len(data))]
    dropped = ['casual', 'cnt', 'datetime', 'registered']
    choice = np.random.choice(lis, size=15000, replace=False)

    train_x, test_x = data.iloc[choice], data.drop(choice)
    train_y, test_y = train_x['casual'], test_x['casual']
    
    train_x, test_x = train_x.drop(columns=dropped), test_x.drop(columns=dropped)

    return train_x, test_x, train_y, test_y

def make_depth_plot(X, y, n, key):
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
    # dtr, rfr = lis

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
    pass
    train_out, test_out = get_classification_frames()
    regress_df = get_regression_frame()
    train_x, test_x, train_y, test_y = get_regression_X_and_y(regress_df)


if  __name__ == "__main__":
    main()