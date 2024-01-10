import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

fonts = fm.findSystemFonts()

for font in fonts:
    fm.fontManager.addfont(font)

font_size = 11
font_family = 'Times New Roman'
plot_style = 'seaborn-v0_8-paper'
# plot_style = 'fivethirtyeight'

plt.style.use(plot_style)
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['font.family'] = font_family
# plt.rcParams['font.size'] = font_size
# plt.rcParams['axes.labelsize'] = font_size
# plt.rcParams['xtick.labelsize'] = font_size
# plt.rcParams['ytick.labelsize'] = font_size
# plt.rcParams['legend.fontsize'] = font_size
# plt.rcParams['figure.titlesize'] = font_size
# plt.rcParams['axes.titlesize'] = font_size + 1
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['figure.constrained_layout.use'] = True


def engineer_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for the change in values over time.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data to engineer features for.

    Returns
    -------
    df : pandas.DataFrame
        The data with the new features added.

    Notes
    -----
    - The features are the change in values after project implementation.
    - The features are calculated for each indicator with 'before' and 'after'.

    Example
    -------
    >>> df = engineer_change_features(df)

    """
    dataframe = df.copy()

    # Create a list of the columns to use
    for col in dataframe.columns:
        if col.endswith('_before') or col.endswith('_after'):
            base_col = col.rsplit('_', 1)[0]
            before_col = base_col + '_before'
            after_col = base_col + '_after'
            if before_col in dataframe.columns and after_col in dataframe.columns:
                change_col = base_col + '_change'
                dataframe[change_col] = dataframe[after_col] - dataframe[before_col]
                change_col_index = max(dataframe.columns.get_loc(before_col), dataframe.columns.get_loc(after_col)) + 1
                change_col_data = dataframe[change_col]
                dataframe.drop(columns=[change_col], inplace=True)
                dataframe.insert(change_col_index, change_col, change_col_data)
    return dataframe


ordinal_map = {'Strongly Agree': 2,
               'Agree': 1,
               'Neutral': 0,
               'Disagree': -1,
               'Strongly Disagree': -2}

def create_composite_feature(df, features, new_feature_name,
                             method='mean', weights=None, drop_features=False,
                             ordinal_map=ordinal_map):
    """
    Create a composite feature from a list of features.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe containing the features.
    features : list
        The list of features to combine.
    new_feature_name : str
        The name of the new feature.
    drop_features : bool, default False
        Whether to drop the features used to create the new feature.
    method : str, default 'mean'
        The method to use to combine the features. Options are 'mean', 'sum',
        'weighted_mean', 'weighted_sum', 'interaction', and 'pca'.
    weights : list, default None
        The weights to use for the weighted_mean and weighted_sum methods.
    ordinal_map : dict, default None
        The mapping to use for ordinal features.

    Returns
    -------
    dataframe : pandas.DataFrame
        The dataframe with the new feature added.
    """
    data = df.copy()

    if ordinal_map is not None:
        dataframe = data.replace(ordinal_map)
    else:
        dataframe = data

    # if new_feature_name in dataframe.columns:
    #     raise ValueError(f'Feature {new_feature_name} already exists.')
    #
    # if not all(feature in dataframe.columns for feature in features):
    #     raise ValueError('Not all features in features exist in dataframe.')

    if method == 'mean':
        dataframe[new_feature_name] = dataframe[features].mean(axis=1)
    #
    # elif method == 'sum':
    #     dataframe[new_feature_name] = dataframe[features].sum(axis=1)
    #
    # elif method == 'weighted_mean':
    #     if weights is None:
    #         raise ValueError('Weights must be provided for weighted_mean method.')
    #     dataframe[new_feature_name] = np.average(dataframe[features], axis=1, weights=weights)
    #
    # elif method == 'weighted_sum':
    #     if weights is None:
    #         raise ValueError('Weights must be provided for weighted_sum method.')
    #     dataframe[new_feature_name] = np.dot(dataframe[features], weights)
    #
    # elif method == 'interaction':
    #     dataframe[new_feature_name] = dataframe[features].prod(axis=1)
    #
    # elif method == 'pca':
    #     dataframe[features] = StandardScaler().fit_transform(dataframe[features])
    #
    #     pca = PCA(n_components=1)
    #
    #     dataframe[new_feature_name] = pca.fit_transform(dataframe[features])
    #
    # if drop_features:
    #     dataframe.drop(columns=features, inplace=True)

    return dataframe


def transform_features(df, features,
                       treat_outliers='drop', treat_skewness=None):
    """
    Transform features in a dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe containing the features.

    features : list
        The list of features to transform.

    treat_outliers : any, default None
        How to treat outliers. If None, no treatment is applied.

    treat_skewness : any, default None
        How to treat skewness. If None, no treatment is applied.

    Returns
    -------
    dataframe : pandas.DataFrame
        The dataframe with the transformed features.
    """
    dataframe = df.copy()

    if not all(feature in dataframe.columns for feature in features):
        raise ValueError('Not all features in features exist in dataframe.')

    if features is None:
        features = dataframe.columns.tolist()

    if not isinstance(features, list):
        features = list(features)

    for feature in features:

        if treat_outliers is not None:
            outliers = dataframe[(np.abs(stats.zscore(dataframe[feature])) > 3)]

            if treat_outliers == 'drop':
                dataframe.drop(outliers.index, inplace=True)

            elif treat_outliers == 'mean':
                dataframe[feature] = np.where(np.abs(stats.zscore(dataframe[feature])) > 3,
                                              dataframe[feature].mean(),
                                              dataframe[feature])

            elif treat_outliers == 'median':
                dataframe[feature] = np.where(np.abs(stats.zscore(dataframe[feature])) > 3,
                                              dataframe[feature].median(),
                                              dataframe[feature])

            elif treat_outliers == 'winsorize':
                dataframe[feature] = stats.mstats.winsorize(dataframe[feature], limits=0.03)

        if treat_skewness is not None:
            const = np.abs(dataframe[feature].min()) + 1

            if treat_skewness == 'cuberoot':
                dataframe[f'{feature} (cbrt)'] = np.cbrt(dataframe[feature])

            elif treat_skewness == 'sqrt':
                dataframe[f'{feature} (sqrt)'] = np.sqrt(dataframe[feature] + const)

            elif treat_skewness == 'log':
                dataframe[f'{feature} (log)'] = np.log(dataframe[feature] + const)

            elif treat_skewness == 'boxcox':
                dataframe[f'{feature} (boxcox)'] = stats.boxcox(dataframe[feature] + const)[0]

    return dataframe
