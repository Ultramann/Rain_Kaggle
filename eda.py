import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rf_imputation import impute_missing_rf
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score

def load_data(n_rows=10000):
    chunker = pd.read_csv('data/train.csv', chunksize=n_rows)
    return chunker, chunker.get_chunk()


def check_hists(df, other=None):
    def get_good_data(df, column):
        values = df[column].values
        return values[np.isfinite(values)]
    columns = (df.columns[1:]) - set('Ref')
    other_frame = isinstance(other, pd.core.frame.DataFrame) 
    for column in columns:
        print df[column].describe()
        plt.hist(get_good_data(df, column), normed=other_frame, bins=50)
        if other_frame:
            plt.hist(get_good_data(other, column), alpha=0.5, normed=True, bins=50)
        plt.title(column)
        raw_input()
        plt.close()


def run_model(df):
    X = df.drop('Expected', axis=1).values
    y = df.Expected.values
    rf = RandomForestRegressor(n_estimators=100, n_jobs=4)
    cvs = cross_val_score(rf, X, y, 'mean_absolute_error', n_jobs=4)
    return cvs
    

if __name__ == '__main__':
    n_rows = 5*10**5
    chunker, df = load_data(n_rows)
    test = df.head(80000)
    imputed_df, changes = impute_missing_rf(test, 10)
    #cross_val = run_model(df)
