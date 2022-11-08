
"""

Description:

The optimal number of features from the nested cv analysis for the feature selection methods 
that yield an explicit ranking (f-test with fdr, mig, pearson correlation) are used here to
construct models on the whole training data i.e. all Dundein samples - using the best discovered
hyperparameters.

These are then used to predict on the 2 external test (EXTEND and TWIN) data sets to assess
generalisation ability.

Author: Trevor Doherty

"""

# Import libraries
import sys
from feature_selection import *
from get_data import *
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lars, PassiveAggressiveRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import time
from utils import *

# Create class instances
feat_sel = feature_selection()
utils = util_funcs()


def custom_scoring_function(y_true, y_pred):
    """Scoring function that calculates the Pearson correlation between predicted and actual."""
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def elastic_net_full_training_and_prediction(X, y, X_train, X_test_extend, X_test_twin, part):
    # Perform 10-fold CV using GroupKFold to ensure paired samples within a single fold 
    groups = np.array(X['snum'])
    cv = GroupKFold(n_splits=10)
    np.random.seed(0)
    model = ElasticNet(random_state=0, max_iter=100000)
    param_combos, parameter_grid = utils.get_parameter_combinations('elastic_net')
    #scorer = make_scorer(custom_scoring_function, greater_is_better=True)
    search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1, verbose=2)
    #search = GridSearchCV(model, parameter_grid, scoring=scorer, cv=cv, refit=True, n_jobs=-1, verbose=2)
    #search = RandomizedSearchCV(model, parameter_grid, scoring='neg_mean_squared_error', n_iter=50, cv=cv, refit=True, n_jobs=-1, verbose=2)
    result = search.fit(X_train, y, groups)
    best_model = result.best_estimator_
    pred_extend = best_model.predict(X_test_extend)
    pred_twin = best_model.predict(X_test_twin)
    model_coefs = pd.DataFrame ([X_train.columns, best_model.coef_]).T.rename(columns={0:'Feature Names', 1:'Coefficients'})
    model_coefs.loc[-1] = ['Intercept', best_model.intercept_]
    model_coefs.to_csv('/home/ICTDOMAIN/d18129068/feature_selection_paper/results/results_external_test_sets/model_coefs_' + part + '.csv')
    # joblib.dump(best_model, '/home/ICTDOMAIN/d18129068/feature_selection_paper/results/results_external_test_sets/model_' + part + '.sav')
    return pred_extend, pred_twin


def elastic_net_full_training_and_prediction_with_scaling(X, y, X_train, X_test_extend, X_test_twin):
    # Perform 10-fold CV using GroupKFold to ensure paired samples within a single fold
    groups = np.array(X['snum'])
    cv = GroupKFold(n_splits=10)
    model = ElasticNet(random_state=0, max_iter=100000)
    param_combos, parameter_grid = utils.get_parameter_combinations('elastic_net')
    search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1, verbose=2)

    # Apply z-score scaling to X_train and transform X_test, also convert y to z-scores
    scaler_x = StandardScaler(); scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train); X_test_extend = scaler_x.transform(X_test_extend); X_test_twin = scaler_x.transform(X_test_twin)
    y = scaler_y.fit_transform(np.array(y).reshape(-1, 1))
    result = search.fit(X_train, y, groups)
    best_model = result.best_estimator_
    pred_extend = best_model.predict(X_test_extend)
    pred_twin = best_model.predict(X_test_twin)
    return pred_extend, pred_twin, scaler_y


def elastic_net_full_training_and_prediction_with_separate_scaling(X, y, X_train, X_test_extend, X_test_twin):
    # Perform 10-fold CV using GroupKFold to ensure paired samples within a single fold
    groups = np.array(X['snum'])
    cv = GroupKFold(n_splits=10)
    model = ElasticNet(random_state=0, max_iter=100000)
    param_combos, parameter_grid = utils.get_parameter_combinations('elastic_net')
    search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1, verbose=2)

    # Apply z-score scaling to X_train and transform X_test, also convert y to z-scores
    scaler_dunedin = StandardScaler(); y = scaler_dunedin.fit_transform(np.array(y).reshape(-1, 1))
    scaler_extend = StandardScaler(); y_scaled_extend = scaler_extend.fit_transform(np.array(y_test_extend).reshape(-1, 1))
    scaler_twin = StandardScaler();  y_scaled_twin = scaler_twin.fit_transform(np.array(y_test_twin).reshape(-1, 1))

    result = search.fit(X_train, y, groups)
    best_model = result.best_estimator_
    pred_extend = best_model.predict(X_test_extend)
    pred_twin = best_model.predict(X_test_twin)
    return pred_extend, pred_twin, y_scaled_extend, y_scaled_twin


def lars_full_training_and_prediction(X, y, X_train, X_test_extend, X_test_twin, part):
    # Perform 10-fold CV using GroupKFold to ensure paired samples within a single fold
    groups = np.array(X['snum'])
    cv = GroupKFold(n_splits=10)
    np.random.seed(0)
    model = Lars(random_state=0)
    parameter_grid = {'n_nonzero_coefs':[10, 100, 200, 400, 800, 1600, 3200, 6499]}
    search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1, verbose=2)
    result = search.fit(X_train, y, groups)
    best_model = result.best_estimator_
    pred_extend = best_model.predict(X_test_extend)
    pred_twin = best_model.predict(X_test_twin)
    model_coefs = pd.DataFrame ([X_train.columns, best_model.coef_]).T.rename(columns={0:'Feature Names', 1:'Coefficients'})
    model_coefs.loc[-1] = ['Intercept', best_model.intercept_]
    model_coefs.to_csv('/home/ICTDOMAIN/d18129068/feature_selection_paper/results/results_external_test_sets/model_coefs_' + part + '.csv')
    return pred_extend, pred_twin


def SVR_full_training_and_prediction(X, y, X_train, X_test_extend, X_test_twin, part):
    # Perform 10-fold CV using GroupKFold to ensure paired samples within a single fold
    groups = np.array(X['snum'])
    cv = GroupKFold(n_splits=10)
    np.random.seed(0)
    model = SVR()
    param_combos, parameter_grid = utils.get_parameter_combinations('svr')
    search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1, verbose=2)
    result = search.fit(X_train, y, groups)
    best_model = result.best_estimator_
    pred_extend = best_model.predict(X_test_extend)
    pred_twin = best_model.predict(X_test_twin)
    model_coefs = pd.DataFrame ([X_train.columns, best_model.coef_[0]]).T.rename(columns={0:'Feature Names', 1:'Coefficients'})
    model_coefs.loc[-1] = ['Intercept', best_model.intercept_[0]]
    model_coefs.to_csv('/home/ICTDOMAIN/d18129068/feature_selection_paper/results/results_external_test_sets/model_coefs_' + part + '.csv')
    return pred_extend, pred_twin


def mlp_full_training_and_prediction(X, y, X_train, X_test_extend, X_test_twin):
    # Perform 10-fold CV using GroupKFold to ensure paired samples within a single fold
    groups = np.array(X['snum'])
    cv = GroupKFold(n_splits=10)
    np.random.seed(0)
    model = MLPRegressor(random_state=0, early_stopping=True)
    param_combos, parameter_grid = utils.get_parameter_combinations('mlp')
    search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1, verbose=2)
    result = search.fit(X_train, y, groups)
    best_model = result.best_estimator_
    pred_extend = best_model.predict(X_test_extend)
    pred_twin = best_model.predict(X_test_twin)
    return pred_extend, pred_twin


def pls_full_training_and_prediction(X, y, X_train, X_test_extend, X_test_twin, part):
    # Perform 10-fold CV using GroupKFold to ensure paired samples within a single fold
    groups = np.array(X['snum'])
    cv = GroupKFold(n_splits=10)
    np.random.seed(0)
    model = PLSRegression()
    parameter_grid = {'n_components':[1, 2, 5, 10, 20, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1250, 1500]}
    search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1, verbose=2)
    result = search.fit(X_train, y, groups)
    best_model = result.best_estimator_
    pred_extend = best_model.predict(X_test_extend).flatten()
    pred_twin = best_model.predict(X_test_twin).flatten()
    model_coefs = pd.DataFrame ([X_train.columns, best_model.coef_.flatten()]).T.rename(columns={0:'Feature Names', 1:'Coefficients'})
    model_coefs.loc[-1] = ['Intercept', best_model.y_mean_[0]]
    model_coefs.to_csv('/home/ICTDOMAIN/d18129068/feature_selection_paper/results/results_external_test_sets/model_coefs_' + part + '.csv')
    return pred_extend, pred_twin


def passive_aggressive_full_training_and_prediction(X, y, X_train, X_test_extend, X_test_twin):
    # Perform 10-fold CV using GroupKFold to ensure paired samples within a single fold
    groups = np.array(X['snum'])
    cv = GroupKFold(n_splits=10)
    model = PassiveAggressiveRegressor(random_state=0, early_stopping=True)
    parameter_grid = {'C':[0.001, 0.01, 0.1, 0.5, 1, 10, 100]}
    search = GridSearchCV(model, parameter_grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1, verbose=2)
    result = search.fit(X_train, y, groups)
    best_model = result.best_estimator_
    pred_extend = best_model.predict(X_test_extend)
    pred_twin = best_model.predict(X_test_twin)
    return pred_extend, pred_twin


def return_metrics_on_external_test_sets(y, yhat, results, method, dataset, runtime):
    """Using predicted TL, calculate performance metrics."""
    mae = mean_absolute_error(y, yhat)
    mape = utils.mean_absolute_percentage_error(y, yhat)
    rmse = mean_squared_error(y, yhat)
    corr, _ = pearsonr(y, yhat)
    print('MAE, MAPE and Correlation: {}, {}, {} for method {} on data set {}'.format(np.round(mae, 3), np.round(mape, 3), np.round(corr, 3), method, dataset))
    results.append((method, dataset, mae, mape, rmse, corr, runtime))
    return results


def save_results_to_file(predictions_extend, predictions_twin, results, part):
    """Save down results and predictions files."""
    predictions_extend.to_csv('../../results/predictions_extend_' + part + '.csv')
    predictions_twin.to_csv('../../results/predictions_twin_' + part + '.csv')
    results_df = pd.DataFrame(results, columns=['Method', 'Dataset', 'MAE', 'MAPE', 'RMSE', 'Correlation', 'Run time'])
    results_df.to_csv('../../results/results_' + part + '.csv')


def main():
    # Dunedin data is the training data set and EXTEND/TWIN are the test data sets
    # Read in training data, test data
    fraction = 1
    # Setting adjusted='No' ensures that the TL for each data set is the TL adjusted by plate ID
    # The input methylation levels are not adjusted - the original values are used.
    betas_status = 'unadjusted_betas_adjusted_tl'
    adjusted = 'No'
    # Get the training and 2 independent data sets
    dunedin_X, dunedin_y, dunedin_26, dunedin_38 = read_dunedin_data_sets(fraction)
    basename_map = get_basename_map(dunedin_y)
    extend_X, extend_y = read_extend_data_sets(adjusted)
    twin_X, twin_y = read_twin_data_sets(adjusted)
    # Transpose, restrict betas to 'cg' columns, remove cols with NaN values, get common CpGs across data sets
    dunedin_df, extend_X, twin_X = structure_data(dunedin_X, dunedin_y, dunedin_26, dunedin_38, extend_X, extend_y, twin_X, twin_y, basename_map)
    # Separate the X and y elements of dunedin_df
    y = dunedin_df.iloc[:, -2]; X = dunedin_df.drop('Telomere Length', axis=1)

    results = []
    predictions_extend = pd.DataFrame(); predictions_twin = pd.DataFrame()
    start = time.time()

    
    # Build model 1: PCA
    # Fit PCA on Dunedin data and transform test data sets
    # EXTEND
    X_train, X_test_extend, top_features = feat_sel.pca(X.iloc[:, :-1], extend_X[X.iloc[:, :-1].columns], 0.99)
    X_train, X_test_twin, top_features = feat_sel.pca(X.iloc[:, :-1], twin_X[X.iloc[:, :-1].columns], 0.99)
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X_train, X_test_extend, X_test_twin, betas_status + 'PCA_0.99')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + 'PCA_0.99', 'EXTEND', time.time() - start)
    predictions_extend['PCA Predicted EXTEND'] = pred_extend; predictions_extend['Actual EXTEND TL'] = extend_y.values
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + 'PCA_0.99', 'TWIN', time.time() - start)
    predictions_twin['PCA Predicted TWIN'] = pred_twin; predictions_twin['Actual TWIN TL'] = twin_y.values
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_PCA_0.99')


    # Build model 2: F-test with FDR (0.01)
    # Rank features using F-test with FDR of 1% on all Dunedin data, fit Elastic Net on the reduced feature set and test on external data
    # EXTEND
    features, ranked_df = feat_sel.apply_f_test_with_fdr(X.iloc[:, :-1], y, 0.01)
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X[ranked_df['Feature Name']], extend_X[ranked_df['Feature Name']], twin_X[ranked_df['Feature Name']], betas_status + '_F-test_0.01')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_F-test_0.01', 'EXTEND', time.time() - start)
    predictions_extend['F-test (0.01) Predicted EXTEND'] = pred_extend; predictions_extend['Actual EXTEND TL2'] = extend_y.values
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_F-test_0.01', 'TWIN', time.time() - start)
    predictions_twin['F-test (0.01) Predicted TWIN'] = pred_twin; predictions_twin['Actual TWIN TL2'] = twin_y.values
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_F_test_0.01')


    # Build model 3: F-test with FDR (0.05)
    # Rank features using F-test with FDR of 5% on all Dunedin data, fit Elastic Net on the reduced feature set and test on external data
    # EXTEND
    features, ranked_df = feat_sel.apply_f_test_with_fdr(X.iloc[:, :-1], y, 0.05)
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X[ranked_df['Feature Name']], extend_X[ranked_df['Feature Name']], twin_X[ranked_df['Feature Name']], betas_status + '_F-test_0.05_recheck')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_F-test_0.05_recheck', 'EXTEND', time.time() - start)
    predictions_extend['F-test (0.05) Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_F-test_0.05_recheck', 'TWIN', time.time() - start)
    predictions_twin['F-test (0.05) Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_F_test_0.05_recheck')


    # Build model 4: Baseline - no prior feature selection - only Elastic Net with its embedded feature selection
    # No prior feature selection - apply elastic net on training data (Dunedin) with CV for tuning, test on external data
    # EXTEND
    print('Running full CV.')
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X.iloc[:, :-1], extend_X[X.iloc[:, :-1].columns], twin_X[X.iloc[:, :-1].columns], betas_status + '_Baseline_090122')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_Baseline_090122', 'EXTEND', time.time() - start)
    predictions_extend['Baseline Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_Baseline_090122', 'TWIN', time.time() - start)
    predictions_twin['Baseline Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_Baseline_090122')


    # Build Model 5: Apply Boostaroota to training data (full Dunedin data) to select feature subset
    # EXTEND
    features, ranked_df = feat_sel.boostaroota(X.iloc[:, :-1], y, 0, 0)
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X[ranked_df['Feature Name']], extend_X[ranked_df['Feature Name']], twin_X[ranked_df['Feature Name']], betas_status + '_Boostaroota_recheck')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_Boostaroota_recheck', 'EXTEND', time.time() - start)
    predictions_extend['Boostaroota Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_Boostaroota_recheck', 'TWIN', time.time() - start)
    predictions_twin['Boostaroota Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_boostaroota_recheck')


    # Build Model 6: Pearson correlation based ranking
    # Apply Pearson correlation-based feature ranking to training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 750 features
    # EXTEND
    features, ranked_df = feat_sel.apply_pearson_r(X.iloc[:, :-1], y, 0)
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:750]], extend_X[ranked_df['Feature Name'][0:750]], twin_X[ranked_df['Feature Name'][0:750]], betas_status + '_pearson_correlation_recheck')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_pearson_correlation_recheck', 'EXTEND', time.time() - start)
    predictions_extend['Pearson Correlation Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_pearson_correlation_recheck', 'TWIN', time.time() - start)
    predictions_twin['Pearson Correlation Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_pearson_correlation_recheck')
    
    
    # Build Model 7: Mutual Informatiom Gain-based ranking
    # Apply MIG correlation-based feature ranking to training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 6500 features
    # EXTEND
    features, ranked_df = feat_sel.mutual_info_gain(X.iloc[:, :-1], y, 0)
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:6500]], extend_X[ranked_df['Feature Name'][0:6500]], twin_X[ranked_df['Feature Name'][0:6500]], betas_status + 'MIG_recheck')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + 'MIG_recheck', 'EXTEND', time.time() - start)
    predictions_extend['MIG Predicted EXTEND'] = pred_extend
    # TWIN
    # Optimal no. of features from nested CV feature selection methods graph was 6500 features
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + 'MIG_recheck', 'TWIN', time.time() - start)
    predictions_twin['MIG Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + 'MIG_recheck')
    

    # Build Model 8: Linear SVR-based feature ranking
    # Apply Linear SVR to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.linear_SVR(X, y, 0, 'unadj_betas_adj_tl_svr')
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:8500]], extend_X[ranked_df['Feature Name'][0:8500]], twin_X[ranked_df['Feature Name'][0:8500]], betas_status + 'Linear SVR')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + 'Linear SVR', 'EXTEND', time.time() - start)
    predictions_extend['Linear SVR Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + 'Linear SVR', 'TWIN', time.time() - start)
    predictions_twin['Linear SVR Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + 'linear_SVR')

    
    # Build Model 9: Random Forest-based feature ranking
    # Apply Random Forest to get feature rankings from training data (full Dunedin data)
    # EXTEND
    # Optimal no. of features from nested CV feature selection methods graph was 3250 features
    features, ranked_df = feat_sel.random_forest(X, y, 0, '_020222')
    pred_extend, pred_twin = elastic_net_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:3250]], extend_X[ranked_df['Feature Name'][0:3250]], twin_X[ranked_df['Feature Name'][0:3250]], betas_status + '_Random_Forest_repeat_020222')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_Random_Forest_repeat_020222', 'EXTEND', time.time() - start)
    predictions_extend['Random Forest Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_Random_Forest_repeat_020222', 'TWIN', time.time() - start)
    predictions_twin['Random Forest Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + 'random_forest_repeat_020222')
    

    # Build Model 10: Pearson correlation based ranking - SCALING OF FEATURES & TARGET
    # Apply Pearson correlation-based feature ranking to training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 750 features
    # EXTEND
    features, ranked_df = feat_sel.apply_pearson_r(X.iloc[:, :-1], y, 0)
    pred_extend, pred_twin, scaler_y = elastic_net_full_training_and_prediction_with_scaling(X, y, X[ranked_df['Feature Name'][0:750]], extend_X[ranked_df['Feature Name'][0:750]], twin_X[ranked_df['Feature Name'][0:750]])

    mae = mean_absolute_error(extend_y, scaler_y.inverse_transform(pred_extend))
    mape = utils.mean_absolute_percentage_error(extend_y, scaler_y.inverse_transform(pred_extend))
    rmse = mean_squared_error(extend_y, scaler_y.inverse_transform(pred_extend))
    corr, _ = pearsonr(extend_y, scaler_y.inverse_transform(pred_extend))
    results.append(('Pearson (scaled X and y)', 'EXTEND', mae, mape, rmse, corr))
    predictions_extend['Pearson Correlation Predicted EXTEND'] = scaler_y.inverse_transform(pred_extend)
    # TWIN
    mae = mean_absolute_error(twin_y, scaler_y.inverse_transform(pred_twin))
    mape = utils.mean_absolute_percentage_error(twin_y, scaler_y.inverse_transform(pred_twin))
    rmse = mean_squared_error(twin_y, scaler_y.inverse_transform(pred_twin))
    corr, _ = pearsonr(twin_y, scaler_y.inverse_transform(pred_twin))
    results.append(('Pearson (scaled X and y)', 'TWIN', mae, mape, rmse, corr))
    predictions_twin['Pearson Correlation Predicted TWIN'] = scaler_y.inverse_transform(pred_twin)
    save_results_to_file(predictions_extend, predictions_twin, results, 'pearson_r_scaling_X_y_replicability')

    
    # Build Model 11: F-test (0.01) - SCALING OF FEATURES & TARGET
    # Apply F-test (0.01) feature selection to training data (full Dunedin data)
    # EXTEND
    features, ranked_df = feat_sel.apply_f_test_with_fdr(X.iloc[:, :-1], y, 0.01)
    pred_extend, pred_twin, scaler_y = elastic_net_full_training_and_prediction_with_scaling(X, y, X[ranked_df['Feature Name']], extend_X[ranked_df['Feature Name']], twin_X[ranked_df['Feature Name']])

    mae = mean_absolute_error(extend_y, scaler_y.inverse_transform(pred_extend))
    mape = utils.mean_absolute_percentage_error(extend_y, scaler_y.inverse_transform(pred_extend))
    rmse = mean_squared_error(extend_y, scaler_y.inverse_transform(pred_extend))
    corr, _ = pearsonr(extend_y, scaler_y.inverse_transform(pred_extend))
    results.append(('F-test (0.01) (scaled X and y)', 'EXTEND', mae, mape, rmse, corr, time.time() - start))
    predictions_extend['F-test (0.01) Predicted EXTEND'] = scaler_y.inverse_transform(pred_extend)
    # TWIN
    mae = mean_absolute_error(twin_y, scaler_y.inverse_transform(pred_twin))
    mape = utils.mean_absolute_percentage_error(twin_y, scaler_y.inverse_transform(pred_twin))
    rmse = mean_squared_error(twin_y, scaler_y.inverse_transform(pred_twin))
    corr, _ = pearsonr(twin_y, scaler_y.inverse_transform(pred_twin))
    results.append(('F-test (0.01) (scaled X and y)', 'TWIN', mae, mape, rmse, corr, time.time() - start))
    predictions_twin['F-test (0.01) Predicted TWIN'] = scaler_y.inverse_transform(pred_twin)
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + 'F_test_1pc_scaling_X_y')
    

    # Build Model 12a: Linear SVR-based feature ranking with Lars instead of EN
    # Apply Linear SVR to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.linear_SVR(X, y, 0, 'extend_twin_eval_full')
    pred_extend, pred_twin = lars_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:8500]], extend_X[ranked_df['Feature Name'][0:8500]], twin_X[ranked_df['Feature Name'][0:8500]])
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, 'Linear SVR with LARS', 'EXTEND')
    predictions_extend['Linear SVR Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, 'Linear SVR with LARS', 'TWIN')
    predictions_twin['Linear SVR Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, 'linear_SVR_with_LARS')

    
    # Build Model 12b: MIG-based feature ranking with Lars instead of EN
    # Apply MIG to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.mutual_info_gain(X.iloc[:, :-1], y, 0)
    pred_extend, pred_twin = lars_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:6500]], extend_X[ranked_df['Feature Name'][0:6500]], twin_X[ranked_df['Feature Name'][0:6500]], betas_status + '_MIG_with_LARS')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_MIG_with_LARS', 'EXTEND', time.time() - start)
    predictions_extend['MIG + LARS Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_MIG_with_LARS', 'TWIN', time.time() - start)
    predictions_twin['MIG + LARS Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_MIG_with_LARS')
    

    # Build Model 13a: Linear SVR-based feature ranking with SVR learner instead of EN
    # Apply Linear SVR to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.linear_SVR(X, y, 0, 'extend_twin_eval_full')
    pred_extend, pred_twin = SVR_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:8500]], extend_X[ranked_df['Feature Name'][0:8500]], twin_X[ranked_df['Feature Name'][0:8500]])
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, 'Linear SVR with SVR Learner', 'EXTEND')
    predictions_extend['Linear SVR Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, 'Linear SVR with SVR Learner', 'TWIN')
    predictions_twin['Linear SVR Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, 'linear_SVR_with_SVR_learner')

    
    # Build Model 13b: MIG-based feature ranking with SVR learner instead of EN
    # Apply MIG to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.mutual_info_gain(X.iloc[:, :-1], y, 0)
    pred_extend, pred_twin = SVR_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:6500]], extend_X[ranked_df['Feature Name'][0:6500]], twin_X[ranked_df['Feature Name'][0:6500]], betas_status + '_MIG_with_SVR_recheck')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_MIG_with_SVR_recheck', 'EXTEND', time.time() - start)
    predictions_extend['MIG + SVR Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_MIG_with_SVR_recheck', 'TWIN', time.time() - start)
    predictions_twin['MIG + SVR Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_MIG_with_SVR_recheck')
    

    # Build Model 14: MIG-based feature ranking with MLP learner instead of EN
    # Apply MIG to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.mutual_info_gain(X.iloc[:, :-1], y, 0)
    pred_extend, pred_twin = mlp_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:6500]], extend_X[ranked_df['Feature Name'][0:6500]], twin_X[ranked_df['Feature Name'][0:6500]], betas_status + '_MIG_with_MLP')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_MIG_with_MLP', 'EXTEND')
    predictions_extend['MIG with MLP Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_MIG_with_MLP', 'TWIN')
    predictions_twin['MIG with MLP Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_MIG_with_MLP')

    
    # Build Model 15a: MIG-based feature ranking with PLS learner instead of EN
    # Apply MIG to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.mutual_info_gain(X.iloc[:, :-1], y, 0)
    pred_extend, pred_twin = pls_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:6500]],
                                                              extend_X[ranked_df['Feature Name'][0:6500]], twin_X[ranked_df['Feature Name'][0:6500]], betas_status + '_MIG_with_PLS')
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, betas_status + '_MIG_with_PLS', 'EXTEND', time.time() - start)
    predictions_extend['MIG with PLS Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, betas_status + '_MIG_with_PLS', 'TWIN', time.time() - start)
    predictions_twin['MIG with PLS Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, betas_status + '_MIG_with_PLS')
    

    # Build Model 15b: Linear SVR-based feature ranking with PLS learner instead of EN
    # Apply Linear SVR to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.linear_SVR(X, y, 0, 'extend_twin_eval_full')
    pred_extend, pred_twin = pls_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:8500]],
                                                              extend_X[ranked_df['Feature Name'][0:8500]], twin_X[ranked_df['Feature Name'][0:8500]])
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, 'Linear SVR with PLS Learner', 'EXTEND')
    predictions_extend['Linear SVR with PLS Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, 'Linear SVR with PLS Learner', 'TWIN')
    predictions_twin['Linear SVR with PLS Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, 'linear_SVR_with_PLS_learner')

    # Build Model 16: Linear SVR-based feature ranking with Passive Aggressive learner instead of EN
    # Apply Linear SVR to get feature rankings from training data (full Dunedin data)
    # Optimal no. of features from nested CV feature selection methods graph was 8500 features
    # EXTEND
    features, ranked_df = feat_sel.linear_SVR(X, y, 0, 'extend_twin_eval_full')
    pred_extend, pred_twin = passive_aggressive_full_training_and_prediction(X, y, X[ranked_df['Feature Name'][0:8500]],
                                                              extend_X[ranked_df['Feature Name'][0:8500]], twin_X[ranked_df['Feature Name'][0:8500]])
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, 'Linear SVR with PLS Learner', 'EXTEND')
    predictions_extend['Linear SVR with PA Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, 'Linear SVR with PA Learner', 'TWIN')
    predictions_twin['Linear SVR with PA Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, 'linear_SVR_with_Passive_Aggressive_learner')


    # Build Model 17: F-test (0.01) - with SVR Learner instead of EN
    # Apply F-test (0.01) feature selection to training data (full Dunedin data)
    # EXTEND
    features, ranked_df = feat_sel.apply_f_test_with_fdr(X.iloc[:, :-1], y, 0.01)
    pred_extend, pred_twin = SVR_full_training_and_prediction(X, y, X[ranked_df['Feature Name']],
                                                                        extend_X[ranked_df['Feature Name']], twin_X[ranked_df['Feature Name']])
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, 'F-test (0.01) with SVR Learner', 'EXTEND')
    predictions_extend['F-test (0.01) with SVR Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, 'F-test (0.01) with SVR Learner', 'TWIN')
    predictions_twin['F-test (0.01) with SVR Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, 'F_test_1pc_with_SVR')


    # Build Model 18: F-test (0.01) - with Passive Aggressive Learner instead of EN
    # Apply F-test (0.01) feature selection to training data (full Dunedin data)
    # EXTEND
    features, ranked_df = feat_sel.apply_f_test_with_fdr(X.iloc[:, :-1], y, 0.01)
    pred_extend, pred_twin = passive_aggressive_full_training_and_prediction(X, y, X[ranked_df['Feature Name']],
                                                                                       extend_X[ranked_df['Feature Name']], twin_X[ranked_df['Feature Name']])
    results = return_metrics_on_external_test_sets(extend_y, pred_extend, results, 'F-test (0.01) with SVR Learner', 'EXTEND')
    predictions_extend['F-test (0.01) with PA Predicted EXTEND'] = pred_extend
    # TWIN
    results = return_metrics_on_external_test_sets(twin_y, pred_twin, results, 'F-test (0.01) with PA Learner', 'TWIN')
    predictions_twin['F-test (0.01) with PA Predicted TWIN'] = pred_twin
    save_results_to_file(predictions_extend, predictions_twin, results, 'F_test_1pc_with_PA')


    # Build Model 19: F-test (0.01) - SCALING OF FEATURES & TARGET
    # Apply F-test (0.01) feature selection to training data (full Dunedin data)
    # EXTEND
    features, ranked_df = feat_sel.apply_f_test_with_fdr(X.iloc[:, :-1], y, 0.01)
    pred_extend, pred_twin, y_scaled_extend, y_scaled_twin = elastic_net_full_training_and_prediction_with_separate_scaling(X, y, X[ranked_df['Feature Name']], extend_X[ranked_df['Feature Name']], twin_X[ranked_df['Feature Name']])

    mae = mean_absolute_error(y_scaled_extend, pred_extend)
    mape = utils.mean_absolute_percentage_error(y_scaled_extend, pred_extend)
    rmse = mean_squared_error(y_scaled_extend, pred_extend)
    corr, _ = pearsonr(y_scaled_extend, pred_extend)
    results.append(('F-test (0.01) (scaled X and y)', 'EXTEND', mae, mape, rmse, corr))
    predictions_extend['F-test (0.01) Predicted EXTEND'] = scaler_y.inverse_transform(pred_extend)
    # TWIN
    mae = mean_absolute_error(y_scaled_twin, pred_twin)
    mape = utils.mean_absolute_percentage_error(y_scaled_twin, pred_twin)
    rmse = mean_squared_error(y_scaled_twin, pred_twin)
    corr, _ = pearsonr(y_scaled_twin, pred_twin)
    results.append(('F-test (0.01) (scaled X and y)', 'TWIN', mae, mape, rmse, corr))
    predictions_twin['F-test (0.01) Predicted TWIN'] = scaler_y.inverse_transform(pred_twin)
    save_results_to_file(predictions_extend, predictions_twin, results, 'F_test_1pc_separate_y_scaled')
    

if __name__ == "__main__":
    main()
