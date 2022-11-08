
"""
Author: Trevor Doherty

Date: Summer 2021

Description: Scripts containing models for testing Elastic Net without feature selection and
with a range of filter feature selection methods. Training data is Dunedin data, independent
test sets are EXTEND and TWIN data sets.

Dunedin data - nested cross-validation is defined for the feature selection + Elatsic Net.
"""

from feature_selection4 import *
from utils4 import *
import joblib
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import SelectKBest, f_regression, SelectFdr
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold, GroupKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skopt import BayesSearchCV
import time


feat_select = feature_selection()
utils_funcs = util_funcs()

class experiments_grid() :
    """Elastic net cross-validation
    """

    def __init__(self):
         print ("")


    def nested_cv(self, X, y, run_results, fs, top_N, parameter_grid, search_type, train_indices, test_indices, ranked_dfs):
        """Apply feature selection and nested cross-validation.

        Nested cross-validation involves parameter tuning on the inner cross-validation
        and testing on the outer cross-validation. Gives representative measure of generalisation.
        Using GroupKFold to ensure that both samples from a subject appear in the same fold/split
        Otherwise, there may be an issue with information leakage across train/test splits in cross-validation/evaluation steps.
        'snum' field denotes the subject number so this is used to ensure groups (subjects) only have samples in one split.

        """
        # configure the cross-validation procedure
        groups_outer = np.array(X['snum'])
        cv_outer = GroupKFold(n_splits=5)
        idx = 0
        for train_ix, test_ix in cv_outer.split(X, y, groups_outer):
            if np.array_equal(train_indices[idx], train_ix):
                print('Train indices consistent')
            if np.array_equal(test_indices[idx], test_ix):
                print('Test indices consistent')

            start = time.time()
            # split data
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            # Get top N features
            # top_feat_cols, ranked_df = feat_select.get_filtered_features(X_train, y_train, fs)
            if fs[0] == 'boostaroota' or fs[0] == 'baseline' or fs[0] == 'f_test_fdr':
                top_feat_cols = ranked_dfs[idx]['Feature Name']
                top_feat_cols = top_feat_cols[top_feat_cols!='snum']
            else:
                top_feat_cols = ranked_dfs[idx]['Feature Name'][0:top_N]
                top_feat_cols = top_feat_cols[top_feat_cols!='snum']

            # configure the inner cross-validation procedure
            groups_inner = np.array(X_train['snum'])
            cv_inner = GroupKFold(n_splits=3)

            # Apply GridSearchCV here along with GroupKFold object
            model = ElasticNet(random_state=0, max_iter=100000)

            #if search_type == 'ElasticNetCV':
            #    search = ElasticNetCV(l1_ratio=parameter_grid['l1_ratio'], n_jobs=-1, random_state=1)
            if search_type == 'GridSearchCV':
                search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', refit=True, n_jobs=-1)
            #if search_type == 'RandomizedSearchCV':
            #    search = RandomizedSearchCV(model, parameter_grid, n_iter=75, scoring='neg_mean_squared_error', cv=cv_inner, refit=True, n_jobs=-1, random_state=1)
            #elif search_type == 'BayesSearchCV':
            #    search = BayesSearchCV(model, search_spaces=parameter_grid, scoring="neg_mean_absolute_error", cv=cv_inner, refit=True, n_jobs=-1, random_state=1)
            #elif search_type == 'HalvingGridSearchCV':
            #    search = HalvingGridSearchCV(model, parameter_grid, scoring="neg_mean_squared_error", min_resources="exhaust", factor=5, cv=cv_inner, refit=True, random_state=1)

            if fs[0] == 'pca':
                X_train, X_test, top_feat_cols = feat_select.pca(X_train, X_test)
            else:
                X_train = X_train[top_feat_cols]; X_test = X_test[top_feat_cols]

            if search_type == 'ElasticNetCV':
                maes, mapes, rmses, corrs, cv_scores, avg_nz_coef, yhats = [], [], [], [], [], [], []
                for tr_idx, tst_idx in cv_inner.split(X_train, y_train, groups_inner):
                    print('Fitting model...')
                    fit_time_start = time.time()
                    result = search.fit(X_train.iloc[tr_idx, :], y_train.iloc[tr_idx])
                    print('Fit time: {}'.format(time.time() - fit_time_start))
                    cv_scores.append(result.score(X_train.iloc[tst_idx, :], y_train.iloc[tst_idx]))
                    refit_model = ElasticNet(l1_ratio=result.l1_ratio_, alpha=result.alpha_, random_state=1)
                    refit_model.fit(X_train, y_train)
                    avg_nz_coef.append(np.count_nonzero(refit_model.coef_))
                    yhat = refit_model.predict(X_test)
                    print(pd.DataFrame(yhat).describe())
                    yhats.append(yhat)
                yhat = np.array(pd.DataFrame(yhats).mean())
                mae = mean_absolute_error(y_test, yhat)
                mape = utils_funcs.mean_absolute_percentage_error(y_test, yhat)
                rmse = mean_squared_error(y_test, yhat)
                corr, _ = pearsonr(y_test, yhat)
                cv_score = np.mean(cv_scores)
                best_params = 'Avg pred of 3 models - no cv_results_ in ElasticNetCV'
                # Log results of test set performance metrics per train/test split run
                best_alpha = 'n/a'; best_l1_ratio = 'n/a'
                nonzero_coeffs = np.mean(avg_nz_coef)
            else:
                print('Fitting model...')
                result = search.fit(X_train, y_train, groups_inner)
                # Get the best performing model fit on the whole training set - this is available because we set refit=True
                best_model = result.best_estimator_
                # evaluate model on the outer folds
                yhat = best_model.predict(X_test)
                print(pd.DataFrame(yhat).describe())
                mae = mean_absolute_error(y_test, yhat)
                mape = utils_funcs.mean_absolute_percentage_error(y_test, yhat)
                rmse = mean_squared_error(y_test, yhat)
                corr, _ = pearsonr(y_test, yhat)
                cv_score = result.best_score_
                best_params = result.best_params_
                best_alpha = result.best_params_['alpha']; best_l1_ratio = result.best_params_['l1_ratio']
                nonzero_coeffs = np.count_nonzero(best_model.coef_)
            # Log results of test set performance metrics per train/test split run
            idx += 1
            # Report progress
            print('>MAE (Pearson for ElatsicNetCV=%.3f, est=%.3f, cfg=%s' % (mae, cv_score, best_params))
            # Store the result
            run_results.append((mae, rmse, mape, corr, cv_score, top_N, len(top_feat_cols), nonzero_coeffs,
                                best_alpha, best_l1_ratio, time.time() - start, 'Run ' + str(idx), fs))
            run_results_df = utils_funcs.save_run_results(run_results, fs, search_type)
        print('Filter: {}, Average Test MAE over the 5 Runs: {}, average MAE over the 5 Runs of 5-fold CV: {}'.format(fs,
            np.round(np.mean(run_results_df['Test MAE Per Run']), 4), np.round(run_results_df['5-fold Inner CV MAE Per Run'].mean(), 4)))
        return run_results




