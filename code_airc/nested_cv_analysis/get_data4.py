
# Library imports
import numpy as np
import pandas as pd
from pdb import set_trace


def read_data_sets():
    """Read in data sets."""
    # ******************************Dunedin data********************************
    dunedin_X= pd.read_pickle('/home/ICTDOMAIN/d18129068/feature_selection/static/Dunedin_betas.pkl')
    dunedin_y = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/Dunedin_pheno.csv', index_col=0)
    
    #dunedin_X= pd.read_pickle('D:/epigenetic_ML/fs_paper/cluster_scripts/static/Dunedin_betas.pkl')
    #dunedin_y = pd.read_csv('D:/epigenetic_ML/fs_paper/cluster_scripts/static/Dunedin_pheno.csv', index_col=0)
    
    # ******************************EXTEND data********************************
    # extend_y, clinical, sample_id_excl = read_labels('TL')
    extend_y, clinical, sample_id_excl, nqr = read_labels('NQR')
    extend_X = read_input(sample_id_excl)
    
    # ******************************TWIN data********************************
    # The next 2 code lines will read in the r file but replaces the column names with numbers
    # Instead the .Rdata file are loaded using RStudio and saved as csv file before loading to Python
    # X = robjects.r['load']("D:\\epigenetic_ML\\dunedin\\exploration\\static\\Telo_twins.Rdata")
    # X = pd.DataFrame(robjects.r['EMMA_pheno_twins'])
    # Read in the twins schizophrenia data (phenotype and DNAm files) from csvs saved in RStudio
    # These are linked by the chip and participant ids.
    twin_y = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/EMMA_pheno_twins.csv', index_col=0)
    twin_X = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/beta_telo_twin.csv', index_col=0)
    twin_nqr = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/EMMA_pheno_twins_nqr.csv', index_col=0)
    nqr_id_map = dict(zip(twin_nqr['X.1'], twin_nqr['RQT/RQS (NQR)']))

    # Transpose betas so that the participants are row-wise and DNAm probes are column-wise
    twin_X = twin_X.T
    twin_X, twin_y = create_DNAm_telomere_matrix(twin_X, twin_y, nqr_id_map)
    return dunedin_X, dunedin_y, extend_X, extend_y, twin_X, twin_y


def create_DNAm_telomere_matrix(X, y, nqr_id_map):
    """Create matrix with DNAm as input variable and telomere length as output variable."""
    # tl_map = dict(zip(y['X.2'], y['TL..kb.'])) 
    #tl_map = dict(zip(y['X.2'], y['telomere_dip']))    
    X['TL'] = X.index.map(nqr_id_map)
    y = X['TL']
    X.drop(columns=['TL'], inplace=True)
    return X, y 

    
def filter_on_cpg_and_ch_features(df):
    """Keep only 'cg' or 'ch' features.

    Excludes 'rs' features etc. 
    """
    # Identify column names containing 'cg' and get list of any that contain >=1 NaN value
    searchfor = ['cg', 'ch']
    meth_cols = df.loc[:, df.columns.str.contains('|'.join(searchfor))].columns
    #cg_cols = [col for col in df.columns if 'cg' in col]
    return meth_cols


def mean_impute_columns_nans(df, cpgs):
    """Mean impute columns that contain NaN values."""
    nan_cg_cols = df[cpgs].columns[df[cpgs].isna().any()].tolist()
    print('No. of CpG columns with at least 1 NaN value: {}'.format(len(nan_cg_cols)))
    print('Filling NaNs with mean of each column...')
    df.fillna(df.mean(), inplace=True)
    #features = list(set(cpgs) - set(nan_cg_cols))
    return df


def filter_data_sets_on_common_cpgs(dunedin_X, extend_X, twin_X,
                                    dunedin_cpgs, extend_cpgs, twin_cpgs):
    """Find the common CpG features and subset data sets on these."""
    common_cpgs = set(dunedin_cpgs) & set(extend_cpgs) & set(twin_cpgs)
    dunedin_X = dunedin_X[common_cpgs]
    extend_X = extend_X[common_cpgs];
    twin_X = twin_X[common_cpgs];
    return dunedin_X, extend_X, twin_X


def structure_data(dunedin_X, dunedin_y, extend_X, extend_y, twin_X, twin_y, fraction):
    """Structure inputs and target."""
    # STRUCTURE DATA
    # Set index back to CpG names as this was reset during read_pickle
    dunedin_X.set_index('Unnamed: 0', inplace=True)
    # Transpose dunedin_X betas
    dunedin_X = dunedin_X.T
    # Subset for testing
    # dunedin_X = dunedin_X.sample(frac=fraction, axis='columns', random_state=42)
    dunedin_X['ID'] = dunedin_y['ID'].values
	# Merge dunedin_X betas and dunedin_y before sorting on participant ids
    dunedin_df = pd.merge(dunedin_X, dunedin_y, on='ID')
    dunedin_df = dunedin_df.sort_values('snum', ascending=True)
    # Get TL from each age column  
    dunedin_df['Telomere Length'] = np.where(dunedin_df['Age'] == 26, dunedin_df['Telomere_pheno_M$TeloBld26'], dunedin_df['Telomere_pheno_M$TelomBld38'])
    # This reduces no. of rows from 1638 to 1615 (NaN at both TL ages)
    dunedin_df = dunedin_df.dropna(subset=['Telomere Length'])
    # Identify column names containing 'cg' and get list of any that contain >=1 NaN value
    dunedin_cpgs = filter_on_cpg_and_ch_features(dunedin_df)
    dunedin_df = mean_impute_columns_nans(dunedin_df, dunedin_cpgs)
    extend_cpgs = filter_on_cpg_and_ch_features(extend_X)
    
    extend_X = mean_impute_columns_nans(extend_X, extend_cpgs)
    twin_cpgs = filter_on_cpg_and_ch_features(twin_X)
    twin_X = mean_impute_columns_nans(twin_X, twin_cpgs)
    # Find CpG features common to data sets
    dunedin_X, extend_X, twin_X = \
        filter_data_sets_on_common_cpgs(dunedin_X, extend_X, twin_X,
                                        dunedin_cpgs, extend_cpgs, twin_cpgs)
    # Add TL, subject number to Dunedin CpGs
    dunedin_features = dunedin_X.columns.tolist()
    dunedin_features.extend(['Telomere Length', 'snum'])
    dunedin_df = dunedin_df[dunedin_features]
    return dunedin_df, extend_X, twin_X


def get_X_y(df):
	"""Split data into train and test sets."""
	# Create train and test sets
	X = df.loc[:, df.columns != 'Telomere Length']
	y = df['Telomere Length']
	return X, y


def create_subsets_for_nested_cross_validation(X, y):
    """Create 5 subsets - 4 for applying CV and 1 for a test set.

    The samples in the subsets have already been ordered and split at indices such that no two
    samples from a subject exist in more than one fold.
    """
    outer_X = [X[0:328], X[328:654], X[654:980], X[980:1306], X[1306:]]
    outer_y = [y[0:328], y[328:654], y[654:980], y[980:1306], y[1306:]]
    # Define 5 CV folds made from 4 of the 5 outer splits - for each permutation
    # Splits 1,2,3,4 + Split 5 is test set
    inner_X1 = [X[0:260], X[260:519], X[519:780], X[780:1039], X[1039:1306]]
    inner_y1 = [y[0:260], y[260:519], y[519:780], y[780:1039], y[1039:1306]]
    # Splits 1,2,3,5 + Split 4 is test set
    inner_X2 = [X[0:260], X[260:519], X[519:780], X.iloc[np.r_[780:980, 1306:1367]], X[1367:]]
    inner_y2 = [y[0:260], y[260:519], y[519:780], y.iloc[np.r_[780:980, 1306:1367]], y[1367:]]
    # Splits 1,2,4,5 + Split 3 is test set
    inner_X3 = [X[0:260], X[260:519], X.iloc[np.r_[519:654, 980:1109]], X.iloc[1109:1367], X[1367:]]
    inner_y3 = [y[0:260], y[260:519], y.iloc[np.r_[519:654, 980:1109]], y.iloc[1109:1367], y[1367:]]
    # Splits 1,3,4,5 + Split 2 is test set
    inner_X4 = [X[0:260], X.iloc[np.r_[260:328, 654:841]], X.iloc[841:1109], X.iloc[1109:1367], X[1367:]]
    inner_y4 = [y[0:260], y.iloc[np.r_[260:328, 654:841]], y.iloc[841:1109], y.iloc[1109:1367], y[1367:]]
    # Splits 2,3,4,5 + Split 1 is test set
    inner_X5 = [X[328:588], X[588:841], X.iloc[841:1109], X.iloc[1109:1367], X[1367:]]
    inner_y5 = [y[328:588], y[588:841], y.iloc[841:1109], y.iloc[1109:1367], y[1367:]]
    inner_X = [pd.concat(inner_X1).reset_index(drop=True), pd.concat(inner_X2).reset_index(drop=True),
               pd.concat(inner_X3).reset_index(drop=True), pd.concat(inner_X4).reset_index(drop=True),
                pd.concat(inner_X5).reset_index(drop=True)]
    inner_y = [pd.concat(inner_y1), pd.concat(inner_y2), pd.concat(inner_y3), pd.concat(inner_y4), pd.concat(inner_y5)]
    return outer_X, outer_y, inner_X, inner_y


def save_results(fs_results):
    """Save results to file."""
    cols = ['FS Method', 'Mean MAE', 'Std. MAE', 'Duration', 'Original Features',
        	'Features - after filters', 'Feature after EN', 'Alpha', 'l1_ratio', 'Description']
    df = pd.DataFrame(fs_results, columns=cols)
    df.to_csv("results.csv")


def read_input(sample_id_excl):
    """Read in methylation data."""
    #X = robjects.r['load']("D:\\epigenetic_ML\\norm_beta_phenotelpath_080816\\static\\normalisedBetas_phenoTelPath08.08.16.Rdata")
    X = pd.read_csv("/home/ICTDOMAIN/d18129068/feature_selection/static/betas.csv", index_col=0)
    # Re-index the dataframe with the methylation sites in the first column before transposing
    # X.set_index('Unnamed: 0', inplace=True)
    X = X.T
    X = X[~X.index.isin(sample_id_excl)]
    return X


def read_labels(y_label):
    """Read in y labels."""
    y = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/pheno_tel_path.csv', index_col=0)
    nqr = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/Raw_Tel_EXTEND.csv', index_col=0)
    #bins = [20, 30, 40, 50, 60, 70]
    #labels = ['20-29', '30-39', '40-49', '50-59', '60-69']
    #y['Age Range'] = pd.cut(y['Age'], bins, labels = labels,include_lowest = True)
    # Drop rows with 'NA' in MeanTelLength and return the SampleID for these.
    # Get additional inputs for model
    nqr_id_map = dict(zip(nqr.index, nqr['RQT/RQS (NQR)']))
    y['NQR'] = y['Sample_ID2'].map(nqr_id_map)

    clinical = y[['SampleID', 'Sample_ID2', 'Age', 'BMI', 'WBC', 'PlasmaBlast', 'CD8pCD28nCD45RAn',
                  'CD8.naive', 'CD4.naive', 'CD8T','CD4T', 'NK', 'Bcell', 'Mono', 'Gran']]
    clinical['SampleID'] = clinical['SampleID'].str[1:]
    y = y[['SampleID', y_label]].set_index('SampleID')
    sample_id_excl = y[y['NQR'].isnull()].index.values
    y = y[~y.index.isin(sample_id_excl)]
    sample_id_excl = [x[1:] for x in sample_id_excl]
    y = y[y_label]
    return y, clinical, sample_id_excl, nqr
