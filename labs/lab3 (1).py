from scipy.optimize import curve_fit
import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt
import statsmodels.api as sm


#read the data from csv to the dataframe
#fit a line to the data

def load_df():
    '''
    Load the csv file into a dataframe 
    Return the dataframe
    Make and return a dataframe that uses the default numbers for the rows 
    and with the column names "years" and "anomalies"
    '''


def fit_line(df):
    '''
    Fit the line using OLs and return the params
    '''


def get_results_frame(df):
    '''
    This dataframe stores the results of the params returned in fit line. 

    Make the row name "linear" and the columns ["a", "b"]
    Insert the params 'x1' and 'const' from fit_line in the data portion of the dataframe
    Return the dataframe
    '''



def make_plot(df, results_frame):
    '''
    Plot the line
    '''


def main():
    '''
    Call the above functions to plot the data
    '''



if __name__ == '__main__':
    main()
    
    