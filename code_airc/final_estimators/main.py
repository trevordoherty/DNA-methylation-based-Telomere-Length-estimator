
"""
Author: Trevor Doherty

Date: Summer/Autumn 2021

Description: Main method
Testing Elastic Net without feature selection and with a range of filter feature selection methods.
Training data is Dunedin study data, independent test sets are EXTEND and TWIN data sets.
Nested CV is conducted with training data set (Dunedin). 

Some of the feature selection techniques are ranking methods e.g. Pearson correlation and plots of
these models are compared across a range of feature set sizes e.g. [0, 250, 500, ..., 20000].
Other feature selection methods automatically select a single subset of features e.g. F-fest with 
FDR.

On assessing the comparative nested CV performance of feature selection techniques with Elastic Net,
these are then used to build models on the whole training data, which are then tested on the
independent data sets (EXTEND and TWIN).
"""

# Library imports
import argparse
from feature_selection import *
from get_data import *
from nested_cv_models import *
from utils import *
import numpy as np
import pandas as pd
import itertools
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression, SelectFdr
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import time
import warnings
warnings.filterwarnings("ignore")


def main():
    """Main method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_search_type", nargs='?', default="check_string_for_empty")
    parser.add_argument("feature_selection", nargs='?', default="check_string_for_empty")
    parser.add_argument("fs_parameter", nargs='?', default="check_string_for_empty")

    args = parser.parse_args()
    search_type = args.parameter_search_type
    fs_method = args.feature_selection
    fs_param = args.fs_parameter

    exp_grid = experiments_grid()
    feat_select = feature_selection()
    utils_funcs = util_funcs()

    start_time = time.time()
    # Get the training and 2 independent data sets
    dunedin_X, dunedin_y, dunedin_26, dunedin_38 = read_dunedin_data_sets(fraction)
    basename_map = get_basename_map(dunedin_y)
    extend_X, extend_y = read_extend_data_sets(adjusted)
    twin_X, twin_y = read_twin_data_sets(adjusted)
    # Transpose, restrict betas to 'cg' columns, remove cols with NaN values, get common CpGs across data sets
    dunedin_df, extend_X, twin_X = structure_data(dunedin_X, dunedin_y, dunedin_26, dunedin_38, extend_X, extend_y, twin_X, twin_y, basename_map)
    # Separate the X and y elements of dunedin_df
    y = dunedin_df.iloc[:, -2]; X = dunedin_df.drop('Telomere Length', axis=1)

    # Get parameter grids
    param_combos, parameter_grid = utils_funcs.get_parameter_combinations('elastic_net')

    # Define feature selection approaches
    run_results = []
    fs_methods = [(fs_method, float(fs_param))]
    """
    ('mutual_info_gain', 0.0001),] , ('f_test_fdr', 0.01), ('f_test_fdr', 0.05), ('baseline', 0.1),
    ('pearson_r', 0.2), ('pearson_r', 0.3), ('variance', 80), ('variance', 90), ('variance', 95) # Zhu et al. used 0.01
    """

    # Precalculate the feature ranking here
    # Have checked if the same train_ix and test_ix are created in each fold of GroupKFold - this
    # ensures either sample from the same subject resides in the same fold -> prevents info leakage.
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
        if fs[0] == 'boostaroota' or fs[0] == 'baseline' or fs[0] == 'f_test_fdr' or fs[0] == 'pca':
            num_feats = [X.shape[1] - 1]
        else:
            num_feats = [50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250,
                         4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250,
                         9500, 9750, 10000, 10250, 10500, 10750, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000, 13250, 13500,
                         13750, 14000, 14250, 14500, 14750, 15000, 15250, 15500, 15750, 16000, 16250, 16500, 16750, 17000, 17250, 17500, 17750,
                         18000, 18250, 18500, 18750, 19000, 19250, 19500, 19750, 20000]
            num_feats = [50, 100, 150]
        for num_feat in num_feats:
            run_results = exp_grid.nested_cv(X, y, run_results, fs, num_feat, parameter_grid, search_type, train_indices, test_indices, ranked_dfs)
    # Can plot the timing graph for search types at this point.
    print('Time taken: {}'.format(time.time() - start_time))

if __name__ == "__main__":
    main()

