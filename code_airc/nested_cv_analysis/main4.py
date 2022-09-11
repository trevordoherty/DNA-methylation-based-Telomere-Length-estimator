
"""
Author: Trevor Doherty

Date: Summer 2021

Description: Main method for Experiments 1 & 2
Testing Elastic Net without feature selection and with a range of filter feature selection methods.
Training data is Dunedin study data, independent test sets are EXTEND and TWIN data sets.

"""

# Library imports
import argparse
from feature_selection4 import *
from get_data4 import *
from models4_gridsearch import *
from utils4 import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import itertools
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression, SelectFdr, VarianceThreshold
from sklearn.linear_model import ElasticNetCV, ElasticNet, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
import time
import warnings
plt.rcParams["figure.figsize"] = (20,6)
warnings.filterwarnings("ignore")

def main():
    """Main method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs='?', default="check_string_for_empty")
    parser.add_argument("parameter_search_type", nargs='?', default="check_string_for_empty")
    args = parser.parse_args()
    search_type = args.parameter_search_type

    fraction = 1
    exp_grid = experiments_grid()
    feat_select = feature_selection()
    utils_funcs = util_funcs()
    
    start_time = time.time()

    # Get the training and 2 independent data sets
    dunedin_X, dunedin_y, extend_X, extend_y, twin_X, twin_y = read_data_sets()
    # Transpose, restrict betas to 'cg' columns, remove cols with NaN values, get common CpGs across data sets
    dunedin_df, extend_X, twin_X = structure_data(dunedin_X, dunedin_y, extend_X, extend_y, twin_X, twin_y, fraction)
    # Separate the X and y elements of dunedin_df
    dunedin_df.reset_index(drop=True, inplace=True)
    y = dunedin_df.iloc[:, -2]; X = dunedin_df.drop('Telomere Length', axis=1) 

    # Get parameter grids
    if args.experiment == 'experiment2':
        param_combos, parameter_grid = utils_funcs.get_parameter_combinations('elastic_net')

    if args.experiment == 'check_string_for_empty':
        print('Please pass experiment to run e.g. For experiment 1 please enter ' + 
         '"python main.py experiment1" in terminal. Other options are experiment2, experiment3 and all.')
    
    elif args.experiment == 'experiment2':
        # Define feature selection approaches
        full_training_results = []; run_results = []
        fs_methods = [('random_forest', 0)] #, ('mutual_info_gain', 0.0001),] #, ('f_test_fdr', 0.01), ('f_test_fdr', 0.05), ('baseline', 0.1), ('pearson_r', 0.2), ('pearson_r', 0.3), ('variance', 80), ('variance', 90), ('variance', 95) # Zhu et al. used 0.01

        # Precalculate the feature ranking here - this saves doing it n x m times where n is the no. of num_feat values and k is K in GroupKFold
        # Have checked if the same train_ix and test_ix are created in each fold of GroupKFold - this should prevent information leakage across train and test
        # within the GroupKFold loop.
        groups_outer = np.array(X['snum'])
        cv_outer = GroupKFold(n_splits=5)
        train_indices = []; test_indices = []; ranked_dfs = []
        for fs in fs_methods:
            idx = 0
            for train_ix, test_ix in cv_outer.split(X, y, groups_outer):
                train_indices.append(train_ix); test_indices.append(test_ix);
                top_feat_cols, ranked_df = feat_select.get_filtered_features(X.iloc[train_ix, :], y[train_ix], fs, idx)
                ranked_dfs.append(ranked_df)
                idx += 1
            if fs[0] == 'boostaroota' or fs[0] == 'baseline':
                num_feats = [X.shape[1] - 1]
            else:
                num_feats = [50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250,
                             4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250,
                             9500, 9750, 10000, 10250, 10500, 10750, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000, 13250, 13500,
                             13750, 14000, 14250, 14500, 14750, 15000, 15250, 15500, 15750, 16000, 16250, 16500, 16750, 17000, 17250, 17500, 17750,
                             18000, 18250, 18500, 18750, 19000, 19250, 19500, 19750, 20000]
                num_feats = [11750, 12000, 12250, 12500, 12750, 13000, 13250, 13500, 13750, 14000, 14250, 14500, 14750, 15000, 15250, 15500, 15750,
                             16000, 16250, 16500, 16750, 17000, 17250, 17500, 17750, 18000, 18250, 18500, 18750, 19000, 19250, 19500, 19750, 20000]
            for num_feat in num_feats:
                run_results = exp_grid.nested_cv(X, y, run_results, fs, num_feat, parameter_grid, search_type, train_indices, test_indices, ranked_dfs)
        # Can plot the timing graph for search types at this point, can also plot the full graph for HalvingSearch results over full feature sets list - if Halving Search is chosen as the best to use...
        print('Time taken: {}'.format(time.time() - start_time))

if __name__ == "__main__":
    main()

