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
    columns = ["years", "anomalies"]
    return pd.read_csv('global_temp_anomalies.csv', index_col = 0, names = columns)


def fit_line(df):
    '''

    '''
    day_array = sm.add_constant(df.index) # necessary to get the intercept
    # above is a new array, so don't have to worry about altering df
    model = sm.OLS(df, day_array)
    results = model.fit()
    print(results)
    return results.params#, results.rsquared, results.mse_resid**0.5, results.fvalue, results.f_pvalue


def get_results_frame(df):
    '''
    This dataframe stores the results of the params returned in fit line. 

    Make the row name "linear" and the columns ["a", "b"]
    '''
    index = ['linear']
    columns = ['a', 'b'] 
    data = []
    params = fit_line(df)
    data.append([params['x1'], params['const']])
    return pd.DataFrame(data, index, columns)


def make_plot(df, results_frame):
    '''
    Plot the data 
    '''
    plt.plot(df.index, results_frame.loc['linear', 'a'] * df.index + results_frame.loc['linear', 'b'], label='linear')
    plt.gca().legend()
    plt.show()

def main():
    '''
    Call all the above functions to plot the data
    '''
    df = load_df()
    print(df)
    results_frame = get_results_frame(df)
    print(get_results_frame(df))
    make_plot(df, results_frame)



if __name__ == '__main__':
    main()
    
    