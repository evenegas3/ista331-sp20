import numpy as np
import pandas as pd
import statistics, math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def get_frames():
    data = pd.read_csv('Boston_Housing.csv')
    np.random.seed(95)

    lis = [x for x in range(len(data))]
    choice = np.random.choice(lis, size=100, replace=False)

    train_X, test_x = data.iloc[choice], data.drop(choice)
    train_y, test_y = train_X['MEDV'], test_x['MEDV']
    train_X, test_x = train_X.drop(columns='MEDV'), test_x.drop(columns='MEDV')

    return train_X, test_x, train_y, test_y

def best_ridge(train_X, train_y, alpha_vals):
    ridge = Ridge()
    parameters = {'alpha': alpha_vals}
    ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)
    ridge_regressor.fit(train_X, train_y)

    best_alpha = ridge_regressor.best_params_
    best_score = ridge_regressor.best_score_

    return best_alpha, best_score
    # rmse = np.sqrt(np.sum((y_predict - test_y) ** 2) / len(test_y))


def best_lasso(training_X, training_y, alpha_vals):
    model = Lasso(alpha=0.05)
    score = cross_val_score(model, training_X, training_y, cv=5, scoring='neg_mean_squared_error')
    # print(score)

def best_net(training_X, training_y, alpha_vals):
    l = []
    model = ElasticNet(alpha=0.5)
    score = cross_val_score(model, training_X, training_y, cv=5, scoring='neg_mean_squared_error')
    # print(score)

def print_ridge(best_ridge_alpha, train_X, train_y, test_x, test_y):
    model = Ridge(alpha=best_ridge_alpha['alpha'])
    model.fit(train_X, train_y)
    y_predict = model.predict(test_x)
    rmse = np.sqrt(np.sum((y_predict - test_y) ** 2) / len(test_y))

    print("""
    Ridge
    Best alpha value: {}
    RMSE: {}
    """.format(best_ridge_alpha['alpha'], rmse))

def least_square(alpha_vals, train_X, train_y, test_x, test_y):
    model = LinearRegression()
    parameters = {'alpha': alpha_vals}
    sq_regressor = GridSearchCV(model, parameters,scoring='neg_mean_squared_error', cv=5)
    sq_regressor.fit(train_X, train_y)

    bestp = sq_regressor.best_params_
    bests = sq_regressor.best_score_

    print(bestp, bests)
    # model = Ridge(alpha=best_ridge_alpha['alpha'])
    # model.fit(train_X, train_y)
    # y_predict = model.predict(test_x)
    # rmse = np.sqrt(np.sum((y_predict - test_y) ** 2) / len(test_y))
    # lin_reg = LinearRegression()
    # MSEs = cross_val_score(lin_reg, train_X, train_y, scoring='neg_mean_squared_error', cv=5)
    # mean_MSE = np.mean(MSEs)
    # print(mean_MSE)
    # lin_reg.fit(train_X, train_y)
    # y_predict = lin_reg.predict(test_x)
    # rmse = np.sqrt(np.sum((y_predict - test_y) ** 2) / len(test_y))
    # print(rmse)


def main():
    train_X, test_x, train_y, test_y = get_frames()
    alpha_vals = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 0.01, 0.05, 0.1, 1, 5, 10, 20]
    best_ridge_alpha, best_ridge_score = best_ridge(train_X, train_y, alpha_vals)
    best_lasso(train_X, train_y, alpha_vals)
    best_net(train_X, train_y, [])

    least_square(alpha_vals, train_X, train_y, test_x, test_y)
    # print_ridge(best_ridge_alpha, train_X, train_y, test_x, test_y)


main()



# def ridge():
#     train_X, test_x, train_y, test_y = get_frames()
#     alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
#     ridge = Ridge()

#     parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

#     ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

#     ridge_regressor.fit(train_X, train_y)

#     print(ridge_regressor.best_params_)
#     print(ridge_regressor.best_score_)

#     print(math.sqrt(ridge_regressor.best_score_))
    # rmse = np.sqrt(np.sum((lis[1].predict(test_x) - test_y) ** 2) / len(test_y))

# ridge()