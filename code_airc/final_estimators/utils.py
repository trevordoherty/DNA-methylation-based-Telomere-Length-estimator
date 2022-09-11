

"""
Author: Trevor Doherty

Date: Summer 2021

Description: Scripts containing assortment of supporting functions for model development.

"""

import itertools
import numpy as np
import pandas as pd

class util_funcs():
    """Class containing a range of supporting functions for model development."""

    def __init__(self):
        print ("")


    def get_parameter_combinations(self, algorithm):
        """Return all parameter combinations of specified parameter lists."""
        if algorithm == 'elastic_net':
            # Get parameter combinations
            parameter_grid = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                       0.25, 0.5, 1, 10, 100, 1000], # some models ran with 0 in the alphas list, but sci-kit recommended against this - may be necessary to add back in if models not replicating
                              'l1_ratio':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                          0.7, 0.8, 0.9, 0.95, 0.99, 1] # 0.0226 - this is from Horvath paper, Docs on EN in sci-kit say l1_ratio <= 0.01 no$
                              }
            param_combos = list(itertools.product(parameter_grid['alpha'],
                                                  parameter_grid['l1_ratio']))
        elif algorithm == 'svr':
            parameter_grid = {'C': [2**-5, 2**-3, 2**-1, 2**0, 2, 2**2,
                                    2**4, 2**6, 2**8, 2**10, 2**12, 2**14],
                              'gamma': [2**-15, 2**-13, 2**-11, 2**-9, 2**-7,
                                        2**-5, 2**-3, 2**-1, 2**0, 2, 2**2, 2**3],
                              'kernel': ['linear']
                              }
            # Get parameter combinations
            param_combos = list(itertools.product(parameter_grid['C'],
                                                  parameter_grid['gamma'],
                                                  parameter_grid['kernel']
                                                  )
                                )
        elif algorithm == 'linearSVR':
            parameter_grid = {'C': [2**-5, 2**-3, 2**-1, 2**0, 2, 2**2,
                                    2**4, 2**6, 2**8, 2**10, 2**12, 2**14]}
            # Get parameter combinations
            param_combos = list(itertools.product(parameter_grid['C']))
        elif algorithm == 'rf':
            parameter_grid = {'max_samples': [0.5, 0.75, 0.99],
                              'max_features': ['auto', 'sqrt', 'log2'],
                              'n_estimators': [10, 50, 100, 200, 500],
                              'max_depth': [5, 10, 20, 50]
                              }
            # Get parameter combinations
            param_combos = list(itertools.product(parameter_grid['max_samples'],
                                                  parameter_grid['max_features'],
                                                  parameter_grid['n_estimators'],
                                                  parameter_grid['max_depth']
                                                  )
                                )
        return param_combos, parameter_grid


    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


    def save_run_results(self, results, fs, search_type):
        """Convert results list to dataframe."""
        cols = ['Test MAE Per Run', 'Test RMSE Per Run', 'Test MAPE Per Run', 'Test Corr Per Run', '5-fold Inner CV MAE Per Run',
                'Original Features', 'Top N Ranked Features', 'Non-zero Coefs after EN', 'Best Alpha', 'Best l1 ratio', 'Time', 'Run', 'Feat Sel']
        results_df = pd.DataFrame(results, columns=cols)
        # fs = ''.join(x for x in str(fs) if x.isalpha())
        results_df.to_csv('/home/ICTDOMAIN/d18129068/feature_selection_paper/results/run_results_' + '_' + search_type + '_' + fs[0] + '_' + str(fs[1]) +  '.csv')
        return results_df
