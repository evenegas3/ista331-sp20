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
    matrix_1_correct = pd.read_pickle('matrix_1_correct.pkl')
    print('want')
    print(matrix_1_correct)
    print('\n')

    cls_tree = DecisionTreeClassifier(max_depth = m_depth)
    # cls_forest = RandomForestClassifier(n_estimators = 500, max_depth = 4)
    cls_forest = DecisionTreeClassifier(max_depth = m_depth)

    cls_tree.fit(train_X, train_y)
    cls_forest.fit(train_X, train_y)

    preds_tree = cls_tree.predict(test_X)
    preds_forest = cls_forest.predict(test_X)

    print("Confusion matrix for tree:")
    print(confusion_matrix(preds_tree, test_y))
    print("\nConfusion matrix for forest:")
    print(confusion_matrix(preds_forest, test_y))
    
def make_depth_plot(x, y, max_depth, model_type):
    averages = np.zeros(max_depth) 
    error = np.zeros(max_depth) 

    for i in range(1, max_depth+1):
        if model_type == 'tree':
            model = DecisionTreeRegressor(max_depth=1)
        if model_type == 'forest':
            model = RandomForestRegressor(n_estimators=25, max_depth=1)
        
        score = cross_val_score(model, x, y, cv = 10, scoring='neg_mean_squared_error')
        averages[i-1] = np.average(score)
        error[i-1] = np.std(score) / np.sqrt(10)

    plt.title('Neg MSE vs. depth, model_type = {}'.format(model_type))
    plt.errorbar(np.range(1, max_depth + 1), averages, yerr=error, color='black', capsize=5)
    plt.xlabel('max_depth')
    plt.ylabel('Negative mean squared error')

    return np.argmax(averages)+1

def main():
    # train, test = get_classification_frames()

    train_X = pd.read_pickle('train_X_correct.pkl')
    train_y = pd.read_pickle('train_y_correct.pkl')
    test_X = pd.read_pickle('test_X_correct.pkl')
    test_y = pd.read_pickle('test_y_correct.pkl')
    
    matrix_1_correct = pd.read_pickle('matrix_1_correct.pkl')
    matrix_3_correct = pd.read_pickle('matrix_3_correct.pkl')

    np.random.seed(23)

    make_and_test_tree(train_X, train_y, test_X, test_y, 1)
    # np.random.seed(23)
    # matrix_3 = make_and_test_tree(train_X, train_y, test_X, test_y, 3)



# main()





# make_depth_plot()