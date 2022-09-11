
"""
Author: Trevor Doherty

Date: Summer 2021

Description: Scripts containing models for Experiments 1 & 2
Testing Elastic Net without feature selection and with a range of filter feature selection methods.
Training data is Dunedin study data, independent test sets are EXTEND and TWIN data sets.

Dunedin data - Nested cross-validation is defined for the Elatsic Net + filter experiments while
standard cross-validation is used for Elastic Net without feature selection.
EXTEND and TWIN - the Elastic Net model trained with no filter and optimal hyper-parameters (experiment 1) 
and the model trained with discovered best filter and optimal hyper-parameters (experiment 2) are used to 
predict the telomere length of samples in the 2 independent data sets. 
"""

from feature_selection4 import *
from utils4 import *
import joblib
import pandas as pd
from pdb import set_trace
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import SelectKBest, f_regression, SelectFdr, VarianceThreshold
from sklearn.linear_model import ElasticNetCV, ElasticNet, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, GroupKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skopt import BayesSearchCV
import time
#from tune_sklearn import TuneGridSearchCV, TuneSearchCV
#from xgboost import XGBRegressor


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
            if fs[0] == 'boostaroota' or fs[0] == 'baseline':
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

            if search_type == 'ElasticNetCV':
                search = ElasticNetCV(l1_ratio=parameter_grid['l1_ratio'], n_jobs=-1, random_state=1)
            if search_type == 'GridSearchCV':
                search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', refit=True, n_jobs=-1)
            if search_type == 'RandomizedSearchCV':
                search = RandomizedSearchCV(model, parameter_grid, n_iter=75, scoring='neg_mean_squared_error', cv=cv_inner, refit=True, n_jobs=-1, random_state=1)
            elif search_type == 'BayesSearchCV': 
                search = BayesSearchCV(model, search_spaces=parameter_grid, scoring="neg_mean_absolute_error", cv=cv_inner, refit=True, n_jobs=-1, random_state=1)
            #elif search_type == 'TuneSearchCV':
            #    search = TuneSearchCV(model, param_distributions=parameter_grid, scoring='neg_mean_absolute_error', early_stopping=True, max_iters=10, cv=cv_inner, search_optimization="bayesian", n_jobs=-1, refit=True, use_gpu=True)
            elif search_type == 'HalvingGridSearchCV':
                search = HalvingGridSearchCV(model, parameter_grid, scoring="neg_mean_squared_error", min_resources="exhaust", factor=5, cv=cv_inner, refit=True, random_state=1)

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
                fit_time_start = time.time()
                result = search.fit(X_train, y_train, groups_inner)
                print('Fit time: {}'.format(time.time() - fit_time_start))
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
                best_alpha = result.best_params_['alpha']; best_l1_ratio = result.best_params_['l1_ratio']#
                nonzero_coeffs = np.count_nonzero(best_model.coef_)
                # Log results of test set performance metrics per train/test split run
            idx += 1
            # Report progress
            print('>MAE (Pearson for ElatsicNetCV=%.3f, est=%.3f, cfg=%s' % (mae, cv_score, best_params))
            # Store the result
            run_results.append((mae, rmse, mape, corr, cv_score, top_N, len(top_feat_cols), nonzero_coeffs,
                                best_alpha, best_l1_ratio, time.time() - start, 'Run ' + str(idx), fs))
            run_results_df = utils_funcs.save_run_results(run_results, fs, search_type, 'experiment2')
        print('Filter: {}, Average Test MAE over the 5 Runs: {}, average MAE over the 5 Runs of 5-fold CV: {}'.format(fs, 
            np.round(np.mean(run_results_df['Test MAE Per Run']), 4), np.round(run_results_df['5-fold Inner CV MAE Per Run'].mean(), 4)))
        return run_results


    def apply_cv_full_training(self, X, y, top_feat_cols, num_feat, parameter_grid, fs, search_type, full_training_results, experiment):
        """Apply 5-fold CV to the full Dunedin data set.

        Uses the 5 pre-defined outer cross-validation test sets from
        experiment 2/3 and GroupKFold to apply 5-fold CV.
        """
        # Add a group number to each of the 5 outer folds for use with GroupKFold
        # configure the cross-validation procedure

        groups = np.array(X['snum'])
        cv = GroupKFold(n_splits=10)

        model = ElasticNet(random_state=0, max_iter=100000)
        search = HalvingGridSearchCV(model, parameter_grid, scoring="neg_mean_absolute_error", n_jobs=-1, min_resources="exhaust", factor=5, cv=cv, refit=True, random_state=0)
        result = search.fit(X.iloc[:, top_feat_cols], y, groups)

        # refit=true was used so model is already trained on the best parameter set - we calculate the training error 
        yhat = result.predict(X.iloc[:, top_feat_cols])
        print(pd.DataFrame(yhat).describe())
        # Get MAE for test fold of CV iteration and log some details of CV stage
        mae = mean_absolute_error(y, yhat)
        mape = utils_funcs.mean_absolute_percentage_error(y, yhat)
        rmse = mean_squared_error(y, yhat, squared=False)
        corr, _ = pearsonr(y, yhat)
        
        full_training_results.append([mae, mape, rmse, corr, len(top_feat_cols), result.best_params_['alpha'], result.best_params_['l1_ratio']])

        full_training_results_df = pd.DataFrame(full_training_results, columns=['Training MAE', 'Training MAPE', 'Training RMSE',
                                                'Training Correlation', 'No. of Features', 'Best Alpha', 'Best l1_ratio'])
        full_training_results_df.to_csv('10_fold_cv_details_for_param_tuning_full_dunedin_data_' + experiment + '_' + fs[0] + '_' + search_type + '.csv')
        return full_training_results_df
    
    
    def build_final_model_on_dunedin_data(self, X, y, top_feat_cols,
                                          parameters, best_filter, experiment
                                          ):
        """Use feature subset, best params to build model on full Dunedin data set.

        This model can then be used to generate predictions on the EXTEND and TWIN datasets.
        Save model down for later use.
        """
        if (experiment == 'experiment1') or (experiment == 'experiment2'):
            model = ElasticNet(alpha=parameters[0], l1_ratio=parameters[1], max_iter=100000, random_state=0)
        elif experiment == 'experiment3':
            model = SVR(C=parameters[0], gamma=parameters[1], kernel=parameters[2])
        elif experiment == 'experiment4':
            model = RandomForestRegressor(max_samples=parameters[0], max_features=parameters[1], n_estimators=parameters[2], max_depth=parameters[3]) 
        elif experiment == 'experiment5':
            model = XGBRegressor(n_estimators=parameters[0], max_depth=parameters[1], eta=parameters[2], subsample=parameters[3], colsample_bytree=parameters[4]) 
        
        if (experiment == 'experiment2') or (experiment == 'experiment3') or (experiment == 'experiment4') or (experiment == 'experiment5'):
            X = X[top_feat_cols]
        model.fit(X, y)
        y_pred = model.predict(X)
        # Get training error on full Dunedin data set - MAE
        mae = mean_absolute_error(y, y_pred)
        mape = utils_funcs.mean_absolute_percentage_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        corr, _ = pearsonr(y, y_pred)
        print('Training error (MAE) on model built using full Dunedin data for testing indep. test sets - {}: {}'.format(experiment, mae))
        print('Training error (MAPE) on model built using full Dunedin data for testing indep. test sets - {}: {}'.format(experiment, mape))
        print('Training error (RMSE) on model built using full Dunedin data for testing indep. test sets - {}: {}'.format(experiment, rmse))
        print('Training error (Correlation Coeff.) on model built using full Dunedin data for testing indep. test sets - {}: {}'.format(experiment, corr))
        training_metrics = \
            pd.DataFrame({'Dundein Training MAE': [mae], 'Dundein Training MAPE': [mape],
                          'Dundein Training RMSE': [rmse], 'Dundein Training Corr': [corr]})
        bf = ''.join(x for x in str(best_filter) if x.isalpha())
        training_metrics.to_csv('dunedin_training_metrics_' + experiment + '_' + bf + '.csv')
        # Save CpG list corresponding to the final features in the Elastic Net model
        if (experiment == 'experiment1') or (experiment == 'experiment2'):
            final_features = pd.DataFrame(X.columns[np.nonzero(model.coef_)[0]], columns=['CpGs from Feature Selection'])
        elif (experiment == 'experiment3') or (experiment == 'experiment4') or (experiment == 'experiment5'):
            final_features = pd.DataFrame(X.columns, columns=['CpGs from Feature Selection'])
        final_features.to_csv('final_CpG_list_' + experiment + '_' + bf + '.csv')        
        # Save the model to disk
        filename = 'saved_model_dunedin_' + experiment + '_' + bf + '.sav'
        joblib.dump(model, filename)
        # load the model from disk
        # loaded_model = joblib.load(filename)
        return model


    def independent_data_set_prediction(self, X, y, model, top_feat_cols, best_filter, experiment, name):
        """Use model trained on Dunedin to predict instances of EXTEND and TWIN data."""
        # Load in EXTEND data
        if (experiment == 'experiment2') or (experiment == 'experiment3') or (experiment == 'experiment4') or (experiment == 'experiment5'):
            X = X[top_feat_cols]
        result = model.score(X, y)
        print(result)
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mape = utils_funcs.mean_absolute_percentage_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        corr, _ = pearsonr(y, y_pred)  
        print('MAE for ' + name + ' data set (model trained on Dunedin): {}'.format(mae))
        print('MAPE for ' + name + ' data set (model trained on Dunedin): {}'.format(mape))
        print('RMSE for ' + name + ' data set (model trained on Dunedin): {}'.format(rmse))
        print('Correlation Coeff. for ' + name + ' data set (model trained on Dunedin): {}'.format(corr))
        indep_metrics = \
            pd.DataFrame({'MAE': [mae], 'MAPE': [mape], 'RMSE': [rmse], 'Corr': [corr]})
        bf = ''.join(x for x in str(best_filter) if x.isalpha())
        indep_metrics.to_csv(name + '_' +  experiment + '_' + bf + '_metrics.csv')
    





