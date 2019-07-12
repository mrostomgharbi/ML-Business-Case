from __future__ import division

import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV

# Example of use  : 
# import model_predict as mp
# params= {
#         'learning_rate': [0.1,0.2,0.4,0.8,1.0],
#         'max_depth': [10,11,13,15,17],
#         'n_estimators': [50, 100],
#     }
# model=mp.Regressor(XGBRegressor(n_jobs = 3),params)
 
class Regressor(BaseEstimator):
 
 
    def __init__(self,regressor,params):
        self.model = BayesSearchCV(estimator=regressor,search_spaces=params,    
        scoring = 'mean_squared_error',
        cv = TimeSeriesSplit(n_splits=3),
        n_jobs = 3,
        n_iter = 10,   
        verbose = 3000,
        refit = True,
        random_state = 42
    )
   
    

    def fit(self, X, y):
        self.model.fit(X, y)
        filename = '/home/mejri/Desktop/TELECOM_PARISTECH_MASTER_X_DATASCIENCE/MACHINE_LEARNING_BUSINESS_CASE/rossmann-store-sales/finalized_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        
    def predict(self, X):
        yres = self.model.predict(X)
        return yres