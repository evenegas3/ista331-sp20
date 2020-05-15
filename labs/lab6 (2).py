from scipy.optimize import curve_fit
import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt
import statsmodels.api as sm




def load_df():
    '''
    Load the csv file into a dataframe 
    Make and return a dataframe that uses the default numbers for the rows 
    and with the column names "years" and "anomalies"
    return the dataframe
    '''
    columns = ["years", "anomalies"]
    return pd.read_csv('global_temp_anomalies.csv', index_col = 0, names = columns)


def fit_model(df, type):
    '''
    this function takes a dataframe and a model type (linear, quadratic, or cubic)
    create a model of the given type and fit it to the data in your dataframe
    return the results, which includes the params along with the r squared, rmse, f value, and p value
    note that sm.OLS will give you the mse, not rmse. how can you derive the rmse from the mse?
    '''
   


def get_results_frame(df, type):
    '''
    this function takes a dataframe and a model type, and returns a dataframe
    This dataframe stores the results of the results returned in fit line/curve. 
    Make the row name the type of model used to to generate the results
    make the columns the the appropriate param labels along with ['R^2', 'RMSE', 'F-stat', 'F-pval']
    the param labels should be 'a', 'b', 'c', 'd'. keep in mind which params are available to you given the model type used.
    '''
    


def make_plot(df, rf, type):
    '''
    this function takes a dataframe, results frame, and model type
    Plot the data from the dataframe as a scatter plot (hint: df.plot's 'linestyle' argument) as well as the line/curve that was fit to the data
    when plotting the line/curve, keep in mind which function you are using to solve for the y-values given the model type used
    '''
   

def main():
    '''
    Call all the above functions to plot the data for a linear, quadratic, and cubic model
    '''
   


if __name__ == '__main__':
    main()
    
    