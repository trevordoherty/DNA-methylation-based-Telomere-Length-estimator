
from utils4 import *
from boostaroota import BoostARoota
import numpy as np
import pandas as pd
from pathlib import Path
from pdb import set_trace
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFdr, VarianceThreshold
#import sklearn_relief as relief
#from skrebate import ReliefF
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
import time
#import sklearn_relief as sr
#from skrebate import ReliefF

util_funcs = util_funcs()

class feature_selection() :
    """Class containing functions relating to Feature Selection."""

    
    def __init__(self):
         print ("")


    def apply_f_test_with_fdr(self, X, y, tol):
        """Return pearson r correlation between each input and the output variable.
            
        Choi et al. used this approach in their Genes journal paper. ie. 
        "We performed F-tests between the beta values of each CpG probe and age and selected CpG sites with
        false discovery rate (FDR, Benjamini–Hochberg procedure [42]) values less than 0.05. Here, the F value
        of the regression analysis is the test result of the null hypothesis that the regression coefficients are
        all zero. Thus, CpG sites that passed the F-test (have a significant FDR of less than 0.05) represented
        CpG features that could have chronological age prediction power[43]. For this F-test, we used the
        “f_regression” function in the feather-selection module provided by “Scikit-learn 0.19.1”"
        However, not sure how the other papers implemented this...did they perform multivariate regressions to
        get the p-values (before adjustment)  - f_regression are univariate p-values. FDR applied afterwards.
        """
        #scores = [pearsonr(X[col], y) for col in X.columns]
        #pearsonr_vals = [x[0] for x in scores]
        #top_feat_indices = [ind for ind, x in enumerate(pearsonr_vals) if x > tol or x < -tol]
        selector = SelectFdr(f_regression, alpha=tol)
        selector.fit(X, y)
        # Get columns to keep and create new dataframe with those only
        cols = selector.get_support(indices=True)
        features = X.iloc[:,cols].columns
        ranked_scores = \
            pd.DataFrame({'Scores': selector.scores_[cols],
                          'Column Number': cols,
                          'Feature Name': features}
                          )
        ranked_df = ranked_scores.sort_values('Scores', ascending=False)
        return features, ranked_df 


    def apply_pearson_r(self, X, y, tol):
        """Return pearson r correlation between each input and the output variable."""
        scores = [pearsonr(X[col], y) for col in X.columns]
        pearsonr_vals = [x[0] for x in scores]
        top_feat_indices = [ind for ind, x in enumerate(pearsonr_vals) if x > tol or x < -tol]
        top_feat_cols = X.iloc[:, top_feat_indices].columns
        ranked_df = \
            pd.DataFrame({'Scores': np.abs(pearsonr_vals),
                          'Feature Name': X.columns}).sort_values('Scores', ascending=False)
        return top_feat_cols, ranked_df


    def variance_threshold(self, X, y, tol):
        """Remove columns with a variance below a specified threshold."""
        sel = VarianceThreshold(threshold=0)
        sel.fit_transform(X)
        threshold = np.percentile(sel.variances_, tol)
        top_feat_indices = [idx for idx, val in enumerate(sel.variances_) if val >= threshold]
        top_feat_cols = X.iloc[:, top_feat_indices].columns
        ranked_df = \
            pd.DataFrame({'Scores': sel.variances_,
                          'Feature Name': X.columns}).sort_values('Scores', ascending=False)
        return top_feat_cols, ranked_df


    def mutual_info_gain(self, X, y, tol):
        """Return MIG scores between each input and the output variable.

        """
        mig_scores = mutual_info_regression(X, y)
        # Create and rank a dataframe of feature names and scores
        ranked_df = pd.DataFrame({'Feature Name': X.columns,
                                  'Scores': mig_scores}).sort_values('Scores', ascending=False)
        print('{}*100 % of MIG scores > 0'.format(np.round(ranked_df[ranked_df['Scores'] > 0].shape[0] / ranked_df.shape[0],3)))
        features = ranked_df['Feature Name']
        return features, ranked_df

    def linear_SVR(self, X, y, tol, idx):
        """Return random forest feature importances for each CpG.

        """
        # configure the inner cross-validation procedure
        #ranked_df = pd.read_csv('rf_importances_fold_' + str(idx) + '.csv')
        #if ranked_df:
        #    pass:
        filepath = '/home/ICTDOMAIN/d18129068/timing_analysis/linearSVC_feature_ranking_per_fold/linearSVC_feature_ranking_fold_' + str(idx) + '.csv'
        file = Path(filepath)
        if file.is_file():
            ranked_df = pd.read_csv(filepath, index_col=0)
            features = ranked_df['Feature Name']
            print('Read in saved linearSVC feature ranking set {}'.format(idx))
        else:
            parameter_grid = {'linear_SVR__C': [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0,
                                                2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7,
                                                2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14],
                             'linear_SVR__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                             'linear_SVR__epsilon': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
                             'linear_SVR__dual': [True, False]
                             }
            parameter_grid = {'linear_SVR__C': [2**-5, 2**-3, 2**-1, 2**0,
                                                2, 2**3, 2**5, 2**7,
                                                2**9, 2**11, 2**13],
                             'linear_SVR__epsilon': [0, 0.25, 0.5, 0.75, 1],
                             }
            groups_inner = np.array(X['snum'])
            cv_inner = GroupKFold(n_splits=3)

            # create pipeline with a scaler
            steps = [('scaler', StandardScaler()), ('linear_SVR', LinearSVR(random_state=0, max_iter=1000, verbose=2))]
            pipe = Pipeline(steps)

            search = RandomizedSearchCV(pipe, parameter_grid, scoring='neg_mean_squared_error', n_iter=75, cv=cv_inner, refit=True, n_jobs=-1, verbose=2)
            fit_time_start = time.time()
            model = search.fit(X.iloc[:, :-1], y, groups_inner)
            print('Fit time: {}'.format(time.time() - fit_time_start))
            feats = {}
            for feature, coef in zip(X.iloc[:, :-1].columns.values,
                                     np.abs(model.best_estimator_.named_steps["linear_SVR"].coef_.flatten())):
                feats[feature] = coef
            ranked_df = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Coefficient'})
            ranked_df.reset_index(inplace=True)
            ranked_df.rename(columns={'index': 'Feature Name'}, inplace=True)
            ranked_df.sort_values('Coefficient', ascending=False, inplace=True)
            print('{} % of LinearSVR scores > 0'.format(np.round(ranked_df[ranked_df['Coefficient'] > 0].shape[0] / ranked_df.shape[0], 3)*100))
            # Save the feature importances for each fold of the nested CV i.e. for each of the 5 inner CV (training)/outer test set pairs
            ranked_df.to_csv('/home/ICTDOMAIN/d18129068/timing_analysis/linearSVC_feature_ranking_per_fold/linearSVC_feature_ranking_fold_' + str(idx) + '.csv')
            features = ranked_df['Feature Name']
        return features, ranked_df

    def boostaroota(self, X, y, tol, idx):
        """Return boostaroota selected features/CpGs.

        """
        # configure the inner cross-validation procedure
        #ranked_df = pd.read_csv('rf_importances_fold_' + str(idx) + '.csv')
        #if ranked_df:
        #    pass:
        filepath = '/home/ICTDOMAIN/d18129068/timing_analysis/boostaroota_feature_rankings_per_fold/feature_ranking_fold_' + str(idx) + '.csv'
        file = Path(filepath)
        if file.is_file():
            ranked_df = pd.read_csv(filepath, index_col=0)
            features = ranked_df['Feature Name']
        else:
            # define Boruta feature selection method
            br = BoostARoota(metric='rmse')

            # find all relevant features - 5 features should be selected
            fit_time_start = time.time()
            br.fit(X.iloc[:, :-1], y)
            print('Fit time: {}'.format(time.time() - fit_time_start))

            ranked_df = pd.DataFrame(br.keep_vars_)
            ranked_df.rename(columns={'feature': 'Feature Name'}, inplace=True)
            print('{} features chosen by BoostARoota'.format(ranked_df.shape[0]))
            features = ranked_df['Feature Name']
            # Save the feature importances for each fold of the nested CV i.e. for each of the 5 inner CV (training)/outer test set pairs
            ranked_df.to_csv('/home/ICTDOMAIN/d18129068/timing_analysis/boostaroota_feature_rankings_per_fold/feature_ranking_fold_' + str(idx) + '.csv')
        return features, ranked_df


    def baseline(self, X, y, tol, idx):
        """Returns full feature set for baseline elastic net model.

        That is, no pre-selection of features is applied before elastic net embedded
        feature selection.
        """
        ranked_df = pd.DataFrame(X.columns, columns=['Feature Name'])
        print('{} features in baseline model'.format(ranked_df.shape[0]))
        features = ranked_df['Feature Name']
        return features, ranked_df


    def relief(self, X, y, nof):
        """Select the top N features using RReliefF.

        RReliefF adapted for continuous target variable - regression.
        """
        start = time.time()
        fs = ReliefF(n_features_to_select=10)
        test = fs.fit_transform(np.array(X.iloc[:, 0:100]), np.array(y))
        end = time.time()
        print('Time for 1000 features from 35pc of data: {}'.format(end - start))
        print(test)

        start = time.time()
        fs = relief.RReliefF(n_features=10)
        #X = fs.fit_transform(np.array(X), np.array(y))
        test = fs.fit_transform(np.array(X.iloc[:, 0:1000]), np.array(y))
        end = time.time()
        print('Time for 100 features from 1000: {}'.format(end - start))
        print(test)

        start = time.time()
        fs = relief.RReliefF(n_features=10)
        #X = fs.fit_transform(np.array(X), np.array(y))
        test = fs.fit_transform(np.array(X.iloc[:, 0:10000]), np.array(y))
        end = time.time()
        print('Time for 1000 features from 10000 {}'.format(end - start))
        print(test)

        set_trace()

        #fs.w_

        #top_feat_indices = [idx for idx, val in enumerate(sel.variances_) if val > threshold]
        #top_feat_cols = X.iloc[:, top_feat_indices].columns
        return top_feat_cols, ranked_df

    def random_forest(self, X, y, tol, idx):
        """Return random forest feature importances for each CpG.

        """
        filepath = '/home/ICTDOMAIN/d18129068/timing_analysis/rf_feature_rankings/rf_feature_ranking_fold_' + str(idx) + '_repeat_020222.csv'
        file = Path(filepath)
        if file.is_file():
            ranked_df = pd.read_csv(filepath, index_col=0)
            features = ranked_df['Feature Name']
        else:
            param_combos, parameter_grid = util_funcs.get_parameter_combinations('rf')
            groups_inner = np.array(X['snum'])
            cv_inner = GroupKFold(n_splits=2)
            model = RandomForestRegressor(random_state=0)
            search = RandomizedSearchCV(model, parameter_grid, scoring='neg_mean_squared_error', cv=cv_inner, refit=True, n_jobs=-1, verbose=2, random_state=0) 
            print('Fitting model...')
            fit_time_start = time.time()
            model = search.fit(X.iloc[:, :-1], y, groups_inner)
            print('Fit time: {}'.format(time.time() - fit_time_start))

            feats = {}
            for feature, importance in zip(X.iloc[:, :-1].columns.values, model.best_estimator_.feature_importances_):
                feats[feature] = importance
            ranked_df = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
            ranked_df.reset_index(inplace=True)
            ranked_df.rename(columns={'index': 'Feature Name'}, inplace=True)
            ranked_df.sort_values('Gini-importance', ascending=False, inplace=True)
            print('{}*100 % of random forest scores > 0'.format(np.round(ranked_df[ranked_df['Gini-importance'] > 0].shape[0] / ranked_df.shape[0],3)))
            features = ranked_df['Feature Name']
            # Save the feature importances for each fold of the nested CV i.e. for each of the 5 inner CV (training)/outer test set pairs
            ranked_df.to_csv('/home/ICTDOMAIN/d18129068/timing_analysis/rf_feature_rankings/rf_feature_ranking_fold_' + str(idx) + '_repeat_020222.csv')
            features = ranked_df['Feature Name']
        return features, ranked_df


    def get_filtered_features(self, X, y, fs, idx):
        """Return subset of filtered features."""
        if fs[0] == 'pearson_r':
            top_feat_cols, ranked_df = self.apply_pearson_r(X, y, fs[1])
        elif fs[0] == 'f_test_fdr':
            top_feat_cols, ranked_df = self.apply_f_test_with_fdr(X, y, fs[1])
        elif fs[0] == 'variance':
            top_feat_cols, ranked_df = self.variance_threshold(X, y, fs[1])
        elif fs[0] == 'mutual_info_gain':
            top_feat_cols, ranked_df = self.mutual_info_gain(X, y, fs[1])
        elif fs[0] == 'relief':
            top_feat_cols, ranked_df = self.relief(X, y, fs[1])
        elif fs[0] == 'LinearSVR':
            top_feat_cols, ranked_df = self.linear_SVR(X, y, fs[1], idx)
        elif fs[0] == 'boostaroota':
            top_feat_cols, ranked_df = self.boostaroota(X, y, fs[1], idx)
        elif fs[0] == 'baseline':
            top_feat_cols, ranked_df = self.baseline(X, y, fs[1], idx)
        elif fs[0] == 'random_forest':
            top_feat_cols, ranked_df = self.random_forest(X, y, fs[1], idx) 
        return top_feat_cols, ranked_df


    def apply_best_filter_to_all_training_data(self, X, y, best_filter):
        """Apply the best identified filter to the whole Dunedin data set."""
        top_feat_cols, ranked_df = self.get_filtered_features(X, y, best_filter)
        return top_feat_cols, ranked_df


    def choose_best_feature_selection_method(self, run_results, fs, experiment):
        """Compare the mean MAE over the 5 runs and choose the best model."""
        run_results_df = util_funcs.save_run_results(run_results, fs, experiment)
        filters_ranked = run_results_df.groupby('Feat Sel')['Test MAE Per Run', 'Test MAPE Per Run',
                                                            'Test RMSE Per Run', 'Test Corr Per Run'].agg(['mean', 'std'])
        fs = ''.join(x for x in str(fs) if x.isalpha())
        filters_ranked.to_csv('average_dunedin_test_set_metrics_by_filter_' + experiment + '_' + fs + '.csv')
        best_filter = filters_ranked.sort_values([('Test MAE Per Run', 'mean')]).index[0]
        best_mae = filters_ranked.sort_values([('Test MAE Per Run', 'mean')])[('Test MAE Per Run', 'mean')].iloc[0]
        best_mape = filters_ranked.sort_values([('Test MAE Per Run', 'mean')])[('Test MAPE Per Run', 'mean')].iloc[0]
        best_rmse = filters_ranked.sort_values([('Test MAE Per Run', 'mean')])[('Test RMSE Per Run', 'mean')].iloc[0]
        best_corr = filters_ranked.sort_values([('Test MAE Per Run', 'mean')])[('Test Corr Per Run', 'mean')].iloc[0]
        
        print('Filter with lowest average MAE (' + '{}) across the 5 Dunedin test splits is: {}'.format(np.round(best_mae, 3), best_filter))
        print('The average MAPE across the 5 Dunedin test splits for this filter is: {}'.format(np.round(best_mape, 3)))
        print('The average RMSE across the 5 Dunedin test splits for this filter is: {}'.format(np.round(best_rmse, 3)))
        print('The average Correlation Coeff. across the 5 Dunedin test splits for this filter is: {}'.format(np.round(best_corr, 3)))
        return run_results_df, best_filter
