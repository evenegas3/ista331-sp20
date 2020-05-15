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
    x = df.index
    const = sm.add_constant(x)
    
    if type == 'quadratic':
        newx = np.column_stack([x, x**2])
        const = sm.add_constant(newx)
        
    if type == 'cubic':
        newx = np.column_stack([x, x**2, x**3])
        const = sm.add_constant(newx)
        
    model = sm.OLS(df, const)
    results = model.fit()
    print(results)
    return results.params, results.rsquared, results.mse_resid**0.5, results.fvalue, results.f_pvalue


def get_results_frame(df, type):
    '''
    this function takes a dataframe and a model type, and returns a dataframe
    This dataframe stores the results of the results returned in fit line/curve. 
    Make the row name the type of model used to to generate the results
    make the columns the the appropriate param labels along with ['R^2', 'RMSE', 'F-stat', 'F-pval']
    the param labels should be 'a', 'b', 'c', 'd'. keep in mind which params are available to you given the model type used.
    '''
    indx = [type]
    results = fit_model(df, type)
    data = []
    
    if type == 'linear':
        data.append([results[0]['x1'], results[0]['const'], results[1], results[2], results[3], results[4]])
        cols = ['a', 'b', 'R^2', 'RMSE', 'F-stat', 'F-pval'] 
    if type == 'quadratic':
        data.append([results[0]['x2'], results[0]['x1'], results[0]['const'], results[1], results[2], results[3], results[4]])
        cols = ['a', 'b', 'c', 'R^2', 'RMSE', 'F-stat', 'F-pval'] 
    if type == 'cubic':
        data.append([results[0]['x3'], results[0]['x2'], results[0]['x1'], results[0]['const'], results[1], results[2], results[3], results[4]])
        cols = ['a', 'b', 'c', 'd', 'R^2', 'RMSE', 'F-stat', 'F-pval']
        
    return pd.DataFrame(data, index = indx, columns = cols)


def make_plot(df, rf, type):
    '''
    this function takes a dataframe, results frame, and model type
    Plot the data from the dataframe as a scatter plot (hint: df.plot's 'linestyle' argument) as well as the line/curve that was fit to the data
    when plotting the line/curve, keep in mind which function you are using to solve for the y-values given the model type used
    '''
    df.plot(linestyle = 'dotted', label = 'data')
    
    if type == 'linear':
        plt.plot(df.index, rf.loc['linear', 'a'] * df.index + rf.loc['linear', 'b'], label='linear')
    if type == 'quadratic':
        plt.plot(df.index,(rf.loc['quadratic','a'] * df.index**2 + rf.loc['quadratic','b'] * df.index + rf.loc['quadratic','c']), label = 'quadratic')
    if type == 'cubic':
        plt.plot(df.index,(rf.loc['cubic','a'] * df.index**3 + rf.loc['cubic','b'] * df.index**2 + rf.loc['cubic','c'] * df.index + rf.loc['cubic','d']), label = 'cubic')
    
    plt.gca().legend()
    plt.show()

def main():
    '''
    Call all the above functions to plot the data for a linear, quadratic, and cubic model
    '''
    df = load_df()
    print(df)
    
    types = ['linear','quadratic','cubic']
    for type in types:
        results_frame = get_results_frame(df, type)
        print(results_frame)
        make_plot(df, results_frame, type)



if __name__ == '__main__':
    main()
    
    