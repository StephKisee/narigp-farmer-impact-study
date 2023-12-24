import pandas as pd


def engineer_change_features(dataframe: pd.DataFrame) -> pd.DataFrame:
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


def create_composite_feature(data, variables, method='mean', weights=None):
    """
    Create a composite feature from a set of variables.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe containing the variables.
    variables : list
        The list of variables to combine.
    method : str, optional
        The method to use to combine the variables ('mean', 'sum', 'weighted_mean', 'weighted_sum', 'interaction', 'pca').
    weights : list, optional
        The list of weights to use for the variables. The default is None (only used for 'weighted_mean' and 'weighted_sum').

    Returns
    -------
    data : pandas.DataFrame
        The dataframe with the new composite feature.

    Examples
    --------
    >>> data = create_composite_feature(data, ['var1', 'var2', 'var3'], method='weighted_mean', weights=[0.5, 0.3, 0.2])
    >>> data = create_composite_feature(data, ['var1', 'var2', 'var3'], method='pca')
    >>> data = create_composite_feature(data, ['var1', 'var2', 'var3'], method='interaction')
    >>> data = create_composite_feature(data, ['var1', 'var2', 'var3'], method='sum')
    """

    # Import library to create a composite feature
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Create a composite feature
    if method == 'mean':
        data['composite'] = data[variables].mean(axis=1)
    elif method == 'sum':
        data[f'composite ({method})'] = data[variables].sum(axis=1)
    elif method == 'weighted_mean':
        data[f'composite ({method})'] = np.average(data[variables], axis=1, weights=weights)
    elif method == 'weighted_sum':
        data[f'composite ({method})'] = np.dot(data[variables], weights)
    elif method == 'interaction':
        data[f'composite ({method})'] = data[variables].prod(axis=1)
    elif method == 'pca':
        # Standardize the data
        data[variables] = StandardScaler().fit_transform(data[variables])

        # Create a PCA object
        pca = PCA(n_components=1)

        # Fit the PCA object
        data[f'composite ({method})'] = pca.fit_transform(data[variables])
    else:
        raise ValueError('Invalid method.')

    return data