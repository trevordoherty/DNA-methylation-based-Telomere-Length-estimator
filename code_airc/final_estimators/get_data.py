"""
Description:

Load data sources - Dunedin, EXTEND and TWIN.
There are 3 sources for Dunedin - betas (with ID basename), phenotype data (ID Basename)
and NRQ TL file (which has Sample_Name that can be linked to Sample_Name in phenotype file).

Some processing is done on the files such as subsetting on either 'cpg' or 'ch' features,
dropping features with any missing values and taking only features that intersect across
the Dunedin, EXTEND and TWIN data sets (to ensure that the same features are available
for the TL estimator for all data sets).

'snum' is passed along with cpg and ch features for Dunedin, to be used as an ID for the
GroupKKold function that will ensure samples from the same subject reside in the same
fold during the cross-validation process.

"""

# Library imports
import numpy as np
import pandas as pd

def read_dunedin_data_sets(fraction):
    """Read in data sets."""
    # ******************************Dunedin data********************************
    dunedin_X = pd.read_pickle('/home/ICTDOMAIN/d18129068/feature_selection/static/Dunedin_betas.pkl')
    dunedin_y = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/Dunedin_pheno.csv', index_col=0)

    # Set index back to CpG names as this was reset during read_pickle
    dunedin_X.set_index(dunedin_X.columns[0], inplace=True)
    # Transpose dunedin_X betas
    dunedin_X = dunedin_X.T
    # Subset for testing
    # dunedin_X = dunedin_X.sample(frac=fraction, axis='columns', random_state=42)
    dunedin_26 = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/plate_adj_TL_Dunedin_26.df.csv', index_col=0)
    dunedin_38 = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/plate_adj_TL_Dunedin_38.df.csv', index_col=0)
    # Add suffix to the UniqueIDNUmber to ensure unique value for mapping basename
    dunedin_26['UniqueIDNumber'] = dunedin_26['UniqueIDNumber'] + '_26'
    dunedin_38['UniqueIDNumber'] = dunedin_38['UniqueIDNumber'] + '_38'
    return dunedin_X, dunedin_y, dunedin_26, dunedin_38


def get_basename_map(dunedin_y):
    """Return map of basenames and Sample_IDs to allow merging with beta files."""
    basename_map = dict(zip(dunedin_y['Sample_Group'], dunedin_y['Basename']))
    return basename_map


def read_extend_data_sets(adjusted):
    """Read in the EXTEND data."""
    # extend_y, clinical, sample_id_excl = read_labels('TL')
    extend_y, clinical, sample_id_excl, nqr = read_labels('NQR')
    extend_X = read_input(sample_id_excl, adjusted)
    if adjusted == 'No':
        extend_X.index = 'X' + extend_X.index
    extend_X = pd.merge(extend_y, extend_X, left_index=True, right_index=True)
    extend_y = extend_X['NQR']
    extend_X.drop(columns=['NQR'], axis=1, inplace=True)
    return extend_X, extend_y


def read_twin_data_sets(adjusted):
    """Read in TWIn data."""
    twin_y = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/EMMA_pheno_twins.csv', index_col=0)
    twin_nqr = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/plate_adj_TL_TWIN.df2.csv')

    nqr_id_map = dict(zip(twin_nqr['X.1'], twin_nqr['Adjusted TL (Plate ID']))
    twin_y['NQR'] = twin_y['X.1'].map(nqr_id_map)

    if adjusted == 'No':
        twin_X = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/beta_telo_twin.csv', index_col=0)
    elif adjusted == 'Yes':
        twin_X = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/adj.Beta_twin.df.csv', index_col=0)
        twin_X.columns = twin_X.columns.str[1:]

    # Transpose betas so that the participants are row-wise and DNAm probes are column-wise
    twin_X = twin_X.T
    twin_X['TL'] = twin_X.index.map(nqr_id_map)
    twin_y = twin_X['TL']
    twin_X.drop(columns=['TL'], inplace=True)
    return twin_X, twin_y


def filter_on_cpg_and_ch_features(df):
    """Keep only 'cg' or 'ch' features.

    Excludes 'rs' features etc. 
    """
    # Identify column names containing 'cg' and get list of any that contain >=1 NaN value
    searchfor = ['cg', 'ch']
    meth_cols = df.loc[:, df.columns.str.contains('|'.join(searchfor))].columns
    return df[meth_cols]


def drop_or_mean_impute_columns_nans(df):
    """Drop/Mean impute columns that contain NaN values."""
    nan_cg_cols = df.columns[df.isna().any()].tolist()
    print('No. of CpG columns with at least 1 NaN value: {}'.format(len(nan_cg_cols)))
    print('Dropping those columns with at least 1 NaN value...')
    df = df.dropna(axis='columns')
    return df


def filter_data_sets_on_common_cpgs(dunedin_X, extend_X, twin_X):
    """Find the common CpG features and subset data sets on these."""
    common_cpgs = set(dunedin_X.columns) & set(extend_X.columns) & set(twin_X.columns)
    dunedin_X = dunedin_X[common_cpgs]
    extend_X = extend_X[common_cpgs];
    twin_X = twin_X[common_cpgs];
    return dunedin_X, extend_X, twin_X


def get_adjusted_tl_map(dunedin_26, dunedin_38, basename_map):
    """Create a mapping for adjusted TL to basename."""
    dunedin_26['Basename'] = dunedin_26['UniqueIDNumber'].map(basename_map)
    dunedin_26.dropna(inplace=True)
    dunedin_38['Basename'] = dunedin_38['UniqueIDNumber'].map(basename_map)
    dunedin_38.dropna(inplace=True)
    dunedin_adj_tl = pd.concat([dunedin_26, dunedin_38], axis=0)
    dunedin_adj_tl_map = dict(zip(dunedin_adj_tl['Basename'],
                                  dunedin_adj_tl['Adjusted TL (Plate ID']))
    return dunedin_adj_tl_map


def structure_data(dunedin_X, dunedin_y, dunedin_26, dunedin_38, extend_X, extend_y, twin_X, twin_y, basename_map):
    """Structure inputs and target."""
    # Map adjusted TL to dunedin_X file
    dunedin_adj_tl_map = get_adjusted_tl_map(dunedin_26, dunedin_38, basename_map)
    dunedin_X['Telomere Length'] = dunedin_X.index.map(dunedin_adj_tl_map)

    # Merge dunedin_X betas and dunedin_y before sorting on participant ids
    dunedin_df = pd.merge(dunedin_X, dunedin_y, left_index=True, right_on='Basename')
    dunedin_df = dunedin_df.sort_values('snum', ascending=True)
    # This reduces no. of rows from 1638 to 1631
    dunedin_df = dunedin_df.dropna(subset=['Telomere Length'])

    # Subset on column names containing 'cg' or 'ch', drop any columns with an NaN
    dunedin_X = filter_on_cpg_and_ch_features(dunedin_X)
    dunedin_X = drop_or_mean_impute_columns_nans(dunedin_X)

    extend_X = filter_on_cpg_and_ch_features(extend_X)
    extend_X = drop_or_mean_impute_columns_nans(extend_X)

    twin_X = filter_on_cpg_and_ch_features(twin_X)
    twin_X = drop_or_mean_impute_columns_nans(twin_X)

    # Find CpG features common to data sets
    dunedin_X, extend_X, twin_X = \
        filter_data_sets_on_common_cpgs(dunedin_X, extend_X, twin_X)

    # Add TL, subject number to Dunedin CpGs
    dunedin_features = dunedin_X.columns.tolist()
    dunedin_features.extend(['Telomere Length', 'snum'])
    dunedin_df = dunedin_df[dunedin_features]
    dunedin_df.reset_index(drop=True, inplace=True)
    return dunedin_df, extend_X, twin_X


def filter_data_sets_on_common_cpgs(dunedin_X, extend_X, twin_X):
    """Find the common CpG features and subset data sets on these."""
    common_cpgs = set(dunedin_X.columns) & set(extend_X.columns) & set(twin_X.columns)
    dunedin_X = dunedin_X[common_cpgs]
    extend_X = extend_X[common_cpgs];
    twin_X = twin_X[common_cpgs];
    return dunedin_X, extend_X, twin_X


def read_input(sample_id_excl, adjusted):
    """Read in methylation data."""
    if adjusted == 'No':
        X = pd.read_csv("/home/ICTDOMAIN/d18129068/feature_selection/static/betas.csv", index_col=0)
    elif adjusted == 'Yes':
        X = pd.read_csv("/home/ICTDOMAIN/d18129068/feature_selection/static/adj_betas_plate_id.csv", index_col=0)
    X = X.T
    X = X[~X.index.isin(sample_id_excl)]
    return X


def read_labels(y_label):
    """Read in y labels."""
    y = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/pheno_tel_path.csv', index_col=0)
    nqr = pd.read_csv('/home/ICTDOMAIN/d18129068/feature_selection/static/plate_adj_TL_EXTEND.df2.csv')

    nqr_id_map = dict(zip(nqr['Sample.ID'], nqr['Adjusted TL (Plate ID']))
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

