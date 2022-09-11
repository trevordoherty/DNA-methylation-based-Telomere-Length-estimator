"""
Description:

This script contains functions for a range of feature selection methods.
These methods are applied in advance of a learning algorithm such as elastic net
regression.

"""

from utils import *
from boostaroota import BoostARoota
import numpy as np
import pandas as pd
from pathlib import Path
from pdb import set_trace
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFdr, VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
import time

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
        filepath = '/home/ICTDOMAIN/d18129068/feature_selection_paper/rankings/linearSVC_feature_ranking_per_fold/linearSVC_feature_ranking_fold_' + str(idx) + '.csv'
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
            groups_inner = np.array(X['snum'])
            cv_inner = GroupKFold(n_splits=3)

            # create pipeline with a scaler
            steps = [('scaler', StandardScaler()), ('linear_SVR', LinearSVR(random_state=0, max_iter=1000, verbose=2))]
            pipe = Pipeline(steps)

            search = RandomizedSearchCV(pipe, parameter_grid, scoring='neg_mean_squared_error', n_iter=75, cv=cv_inner, refit=True, n_jobs=-1, verbose=2)
            print('Fitting model...')
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
            ranked_df.to_csv('/home/ICTDOMAIN/d18129068/feature_selection_paper/rankings/linearSVC_feature_ranking_per_fold/linearSVC_feature_ranking_fold_' + str(idx) + '.csv')
            features = ranked_df['Feature Name']
        return features, ranked_df


    def boostaroota(self, X, y, tol, idx):
        """Return boostaroota selected features/CpGs.

        """
        filepath = '/home/ICTDOMAIN/d18129068/feature_selection_paper/rankings/boostaroota_feature_rankings_per_fold/feature_ranking_fold_' + str(idx) + '.csv'
        file = Path(filepath)
        if file.is_file():
            ranked_df = pd.read_csv(filepath, index_col=0)
            features = ranked_df['Feature Name']
        else:
            # define Boruta feature selection method
            br = BoostARoota(metric='rmse')

            # find all relevant features
            fit_time_start = time.time()
            br.fit(X.iloc[:, :-1], y)
            print('Fit time: {}'.format(time.time() - fit_time_start))

            ranked_df = pd.DataFrame(br.keep_vars_)
            ranked_df.rename(columns={'feature': 'Feature Name'}, inplace=True)
            print('{} features chosen by BoostARoota'.format(ranked_df.shape[0]))
            features = ranked_df['Feature Name']
            # Save the feature importances for each fold of the nested CV i.e. for each of the 5 inner CV (training)/outer test set pairs
            ranked_df.to_csv('/home/ICTDOMAIN/d18129068/feature_selection_paper/rankings/boostaroota_feature_rankings_per_fold/feature_ranking_fold_' + str(idx) + '.csv')
        return features, ranked_df


    def random_forest(self, X, y, tol, idx):
        """Return random forest feature importances for each CpG.

        """
        filepath = '/home/ICTDOMAIN/d18129068/feature_selection_paper/rankings/rf_feature_rankings/rf_feature_ranking_fold_' + str(idx) + '.csv'
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
            ranked_df.to_csv('/home/ICTDOMAIN/d18129068/feature_selection_paper/rankings/rf_feature_rankings/rf_feature_ranking_fold_' + str(idx) + '.csv')
            features = ranked_df['Feature Name']
        return features, ranked_df


    def pca(self, X_train, X_test, pc):
        """Apply PCA."""
        # Apply z-score scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Apply PCA to get components relating to the specified variance
        pca = PCA(pc)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print('The number of principal components corresponding to {}% variance for the training data is: {}'.format(pc*100, pca.n_components_))
        top_feat_cols = pca.singular_values_
        return X_train, X_test, top_feat_cols


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
