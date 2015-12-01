import numpy as np
from time import time
from sys import stdout
from sklearn.ensemble import RandomForestRegressor

def get_cols_with_missing_dict(df):
    '''
    Input:  DataFrame - with columns that have missing values
    Output: Dict - keys: column name, values: np.ndarray of indexes with Nan's
    '''
    missing_idx_gen = ((column, np.isnan(df[column])) for column in df.columns)
    missing_dict = {column: missing_idx for column, missing_idx 
                                        in missing_idx_gen 
                                        if np.any(missing_idx)}
    return missing_dict


def update_column_missing_guesses(best_guess_df, column, missing_idx):
    '''
    Input:  Str - column name
    Output: None
    
    Make random forest regressor model with column as target and remaining columns as input.
    Update predictions for missings in given column.
    '''
    y_train = best_guess_df[column].values
    X_train = best_guess_df.drop(column, axis=1).values
    column_model = RandomForestRegressor(n_estimators=20, n_jobs=4).fit(X_train, y_train)
    X_missing = best_guess_df.drop(column, axis=1).ix[missing_idx, :].values
    missing_guesses = column_model.predict(X_missing) 
    current_guesses = best_guess_df.ix[missing_idx, column].values
    best_guess_df.ix[missing_idx, column] = missing_guesses
    return (((current_guesses - missing_guesses)**2)**.5).sum()
            

def impute_missing_rf(df, iterations=5):
    '''
    Input:  DataFrame (with missing values)
    Output: None

    Starting from mean value of column as guesses for missing values, iteratively make random
    forest model to better guess the missing values.
    '''

    cols_missing_dict = get_cols_with_missing_dict(df)
    best_guess_df = df.fillna(df.mean())
    col_specific_str = '  Finished column {}, time spent: {:.2f}s, '
    time_spent_str = 'average per column: {:.2f}s, cumulative iteration: {:.2f}s'
    column_time_str = col_specific_str + time_spent_str

    changes = []
    for i in range(1, iterations+1):
        print 'Starting iteration {}'.format(i)
        iter_start_time = time()
        column_number = 1
        change = 0
        for column, missing_idx in cols_missing_dict.iteritems():
            column_start_time = time()
            change = update_column_missing_guesses(best_guess_df, column, missing_idx)
            column_time = time() - column_start_time
            cum_time = time() - iter_start_time
            average_column_time = cum_time / column_number
            stdout.flush()
            stdout.write(column_time_str.format(column_number, column_time, 
                                                average_column_time, cum_time) + '  \r')
            column_number += 1
        changes.append(change)
        stdout.flush()
        print '  ' + time_spent_str.capitalize().format(average_column_time, cum_time) + ' '*41 + '\r'

    return best_guess_df, changes
