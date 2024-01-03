
import json

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from plotly.subplots import make_subplots
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor

font_files = fm.findSystemFonts(fontpaths=None, fontext='otf')

for font_file in font_files:
    fm.fontManager.addfont(font_file)

# Setting plot display options
plt.style.use('seaborn-v0_8-paper')
# plt.rcParams['font.family'] = 'Amasis MT W1G'
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Sono'
# plt.rcParams['font.family'] = 'Calisto MT'
# plt.rcParams['font.family'] = 'Sitka'
# plt.rcParams['font.family'] = 'Maiandra GD'
# plt.rcParams['font.family'] = 'Book Antiqua'
# plt.rcParams['font.family'] = 'Palatino Linotype'
# plt.rcParams['font.family'] = 'Rockwell'
# plt.rcParams['font.family'] = 'Georgia'
font_size = 12
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size - 1.5
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['figure.titlesize'] = font_size
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['figure.constrained_layout.use'] = True


def get_varnames(file = '../data/processed/variable_names.json'):

    with open(file, 'r') as f:
        var_names = json.load(f)['variable_names']

    return var_names

names = get_varnames()


def show_df(df: pd.DataFrame, info: bool = True, head: bool = True,
            describe: bool = False, save: bool = False, names: dict = names):
    """
    Show the information about a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to show.
    head : bool, optional
        Whether to show the head of the DataFrame. The default is True.
    describe : bool, optional
        Whether to show the description of the DataFrame. The default is False.
    save : bool, optional
        Whether to save the DataFrame as a CSV file. The default is False.

    Returns
    -------
    None.
    """
    df = df.rename(columns=names)

    if info:
        print("Dataframe Information")
        display(df.info())

    if head:
        print("\nFirst 5 rows")
        display(df.head())

    if describe:
        descriptive_stats = df.describe().T
        descriptive_stats.index.name = "Variable"
        descriptive_stats.rename(columns={"count": "Observations",
                                          "mean": "Mean",
                                          "std": "Standard deviation",
                                          "min": "Minimum",
                                          "25%": "25th percentile",
                                          "50%": "Median",
                                          "75%": "75th percentile",
                                          "max": "Maximum"}, inplace=True)
        descriptive_stats["Observations"] = descriptive_stats["Observations"].astype(int)
        print("Descriptive Statistics")
        display(descriptive_stats.round(2))

        if save:
            descriptive_stats[[
                "Observations",
                "Mean",
                "Standard deviation",
                "Minimum",
                "Maximum"
            ]].to_csv("../results/tables/descriptive_stats.csv", float_format="%.2f", index=True)


def show_dist(df: pd.DataFrame, column: str):
    """
    Show the distribution of a variable in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to show.
    column : str
        The column to show.

    Returns
    -------
    None.
    """
    min = df[column].min()
    max = df[column].max()
    mean = df[column].mean()
    median = df[column].median()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.8, 0.2])

    fig.add_trace(go.Histogram(x=df[column], name=column), row=1, col=1)
    fig.update_layout(template="presentation", font_family="Times New Roman",
                      showlegend=False, title=f"Distribution of {column}")
    fig.update_xaxes(title_text="Frequency", row=1, col=1)
    fig.add_shape(type="line", x0=min, y0=0, x1=min, y1=60,
                  line=dict(color="gray", width=2), row=1, col=1)
    fig.add_shape(type="line", x0=max, y0=0, x1=max, y1=60,
                    line=dict(color="gray", width=2), row=1, col=1)
    fig.add_shape(type="line", x0=mean, y0=0, x1=mean, y1=60,
                    line=dict(color="cyan", width=2), row=1, col=1)
    fig.add_shape(type="line", x0=median, y0=0, x1=median, y1=60,
                    line=dict(color="red", width=2), row=1, col=1)

    fig.add_trace(go.Box(x=df[column], name=column), row=2, col=1)
    fig.update_yaxes(title_text="", row=2, col=1, showticklabels=False)

    fig.update_layout(height=600)
    fig.show()


def missing_by_country(dfs, globals):

    if not isinstance(dfs, list):
        dfs = [dfs]

    for df in dfs:
        df_name = [x for x in globals if globals[x] is df][0]
        print(f"Missing values by country | {df_name}")
        df_show = df.groupby('i').apply(lambda x: x.isna().sum())
        df_show.loc['Total', :] = df_show.sum(axis=0)
        df_show = df_show.astype('int64')
        display(df_show)


def values_by_country(dfs, globals):

    if not isinstance(dfs, list):
        dfs = [dfs]

    for df in dfs:
        df_name = [x for x in globals if globals[x] is df][0]
        print(f"Values by country | {df_name}")
        df_show = df.groupby('i').apply(lambda x: x.nunique())
        df_show.loc['Total', :] = df_show.sum(axis=0)
        df_show = df_show.astype('int64')
        display(df_show)


def time_series_plot(df: pd.DataFrame, df_yr: pd.DataFrame,
                     column: str, ax: plt.Axes, panel: str,
                     names: dict = names):

    sns.lineplot(data=df.reset_index(), x='t', y=column,
                 hue='i', style='i', markers=True,
                 estimator=None, ax=ax)

    ax.plot(df_yr.index, df_yr[column], color='black', linewidth=7.5, alpha=0.2,
            label='Average')

    ax.set_xlabel('Year')
    ax.set_ylabel(f'{names[column]} [%]')

    if column[0] != 'E':
        title = names[column]
        title_end = 'across East Africa'

    else:
        title = names[column].split(' ')[-1].capitalize()
        title_end = 'share of employment'

    ax.set_title(f'Panel{panel}: {title} {title_end} since 1991',
                 fontfamily='Copperplate Gothic Bold',
                 loc='left', pad=10)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(ax.plot([], [], color='black', linewidth=5, alpha=0.25)[0])
    labels.append('Average')

    ax.legend_.remove()

    return ax, handles, labels


def get_i_dict(df: pd.DataFrame, column: str):

    i_dict = {country: i + 1 for i, country in enumerate(df[column].unique())}

    return i_dict


def i_discretizer(df, i_dict, inplace=False, inverse=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(i_dict, dict):
        raise TypeError("i_dict must be a dictionary")

    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a boolean")

    if inverse:
        i_dict = {v: k for k, v in i_dict.items()}

    if 'i' not in df.columns:
        if 'i' not in df.index.names:
            raise ValueError("i must be in the columns or index")
        else:
            df.reset_index(inplace=True)

    if inplace:
        df['i'] = df['i'].map(i_dict)

    else:
        df = df.copy()
        df['i'] = df['i'].map(i_dict)

        return df


def impute(df):
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                               random_state=0)

    df[df.columns] = imputer.fit_transform(df[df.columns].values)

    return df


def transform(df: pd.DataFrame,
              columns: any,
              quantile: any = 0.01,
              drop_outliers: bool = True,
              mean: bool = False,
              median: bool = False,
              winsorize: bool = False,
              cuberoot: bool = False,
              log: bool = False,
              boxcox: bool = False,
              names: dict = names):
    if not isinstance(columns, list):
        columns = [columns]

    if not isinstance(quantile, list):
        quantiles = [quantile, 1 - quantile]
        limits = [quantile, quantile]
    else:
        quantiles = quantile
        limits = quantile

    for col in columns:
        outliers = df[col].quantile(quantiles).values

        if drop_outliers:
            df = df[(df[col] > outliers[0]) & (df[col] < outliers[1])]

        if mean:
            df[col] = np.where(df[col] < outliers[0], df[col].mean(), df[col])
            df[col] = np.where(df[col] > outliers[1], df[col].mean(), df[col])

        if median:
            df[col] = np.where(df[col] < outliers[0], df[col].median(), df[col])
            df[col] = np.where(df[col] > outliers[1], df[col].median(), df[col])

        if winsorize:
            df[col] = stats.mstats.winsorize(df[col], limits=limits)

        if cuberoot:
            df[col] **= (1 / 3)

        const = abs(df[col].min()) + 1

        if log:
            if df[col].min() < 0:
                df[col] += const

            df[col] = np.log(df[col])

            # df.rename(columns={col: col.split()[0] + ' (ln)'}, inplace=True)
            # keys = df.reset_index().columns
            # values = names['variable_names'].values()
            # names['variable_names'] = {k: v for k, v in zip(keys, values)}
            #
            # with open('../data/processed/variable_names.json', 'w') as f:
            #     json.dump(names, f)

        if boxcox:
            if df[col].min() < 0:
                df[col] += const

            df[col], ld = stats.boxcox(df[col])

            # df.rename(columns={col: col.split()[0] + f' (bc_{ld.round(1)})'}, inplace=True)
            # with open('../data/processed/variable_names.json', 'w') as f:
            #     names = json.load(f)
            #     names[col] = col.split()[0] + f' (bc_{ld.round(1)})'
            #     json.dump(names, f)

    return df


def time_series_fig(df: pd.DataFrame, df_yr: pd.DataFrame,
                    columns: any, save: bool = False):
    if not isinstance(columns, list):
        columns = [columns]

    cols = len(columns)

    fig, ax = plt.subplots(1, cols, figsize=(cols * 5, 4), gridspec_kw={'wspace': 0.05})

    if cols == 1:
        fig.set_size_inches(9, 4)
        time_series_plot(df, df_yr, columns[0], ax, ' a')
        handles, labels = ax.get_legend_handles_labels()

    else:
        # Iterate over the columns
        for i in range(cols):
            time_series_plot(df, df_yr, columns[i], ax[i], f' {chr(97 + i)}')

        handles, labels = ax[0].get_legend_handles_labels()

    columns_str = '_'.join(columns)

    fig.legend(handles, labels, loc='lower center',
               ncol=6, bbox_to_anchor=(0.5, -0.1), frameon=False)

    plt.show()

    if save:
        fig.savefig(f'../results/figures/time_series_{columns_str}.png',
                    bbox_inches='tight')


def heterogeneity_fig(df: pd.DataFrame, column: str, names: dict = names, save: bool = False):
    """
    Plot the heterogeneity of GDP growth (annual %) by country and year (change ratio to 2:3)
    """

    df = df.reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4),
                           gridspec_kw={'width_ratios': [3, 4], 'wspace': 0.05}, sharey='row')

    sns.boxplot(data=df, x='i', y=column, width=0.5, ax=ax[0])
    sns.boxplot(data=df, x='t', y=column, ax=ax[1])

    column_name = names[column]

    if column[0] != 'E':
        title = names[column]

    else:
        title = names[column].split(' ')[-1].capitalize()

    title = title + ', heterogeneity across'
    title = 'Heterogeneity across'

    ax[0].axes.xaxis.set_label_text('')
    ax[0].axes.yaxis.set_label_text(column_name + ' [%]')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

    ax[1].axes.xaxis.set_label_text('Year')
    ax[1].axes.yaxis.set_label_text('')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

    ax[0].set_title(f'Panel a: {title} Country',
                    fontfamily='Copperplate Gothic Bold', loc='left', pad=10)
    ax[1].set_title(f'Panel b: {title} Time',
                    fontfamily='Copperplate Gothic Bold', loc='left', pad=10)

    plt.show()

    if save:
        fig.savefig(f'../results/figures/heterogeneity_{column}.png')


def correlation(df: pd.DataFrame, column: str, sort: bool = True,
                specific_plot: bool = True, names: dict = names,
                corr_df: bool = True, save: bool = False):
    c = column

    column = names[column]

    # Compute correlation matrix
    corr = df.rename(columns=names).corr()

    corr.index.name = 'Variable'

    if corr_df:
        print("Correlation Matrix")
        display(corr.round(2))

        if save:
            corr.to_csv('../results/tables/correlation_matrix.csv', float_format="%.2f", index=True)

    # Sort the correlation matrix
    if sort:
        # corr = corr.sort_values(by=column, ascending=False)
        corr = corr.sort_values(
            by=column, axis=0, ascending=False).sort_values(
            by=column, axis=1, ascending=False
        )

    corr = corr.replace(1, np.nan)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.color_palette("viridis", as_cmap=True)

    cols = 1
    panel = ''

    if specific_plot:
        cols = 2
        panel = 'Panel a: '

    # Set up the matplotlib figure
    fig, ax = plt.subplots(1, cols, figsize=(cols * 5, 5), gridspec_kw={'wspace': 0.00})

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, cbar=False,
                annot=True, fmt='.1f', ax=ax[0])

    ax[0].set_title(f'{panel}Correlation Matrix',
                    fontfamily='Copperplate Gothic Bold', loc='left', pad=10)

    if specific_plot:
        sns.barplot(x=corr.index, y=corr[column], ax=ax[1])

        ax[1].set_title(f'Panel b: Correlation with {column}',
                        fontfamily='Copperplate Gothic Bold', loc='left', pad=10)
        # ax[1].yaxis.tick_right()
        # ax[1].yaxis.set_label_position("right")
        # ax[1].spines['left'].set_visible(False)
        # ax[1].spines['right'].set_visible(True)
        ax[1].set_ylabel('')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

    if save:
        plt.savefig(f'../results/figures/correlation_{c}.png')

    # Show the plot
    plt.show()


def scatter_plot(df: pd.DataFrame,
                 x: str,
                 y: str,
                 ax: plt.Axes,
                 panel: str,
                 names: dict = names):
    """Scatter plot of two variables."""
    sns.scatterplot(x=x, y=y, data=df, ax=ax, hue='i', style='i')
    sns.regplot(x=x, y=y, data=df, ax=ax, scatter=False, line_kws={'color': 'gray'})

    x_label = names[x]
    y_label = names[y]

    ax.set_xlabel(f'{x_label} [%]')
    ax.set_ylabel(f'{y_label} [%]')

    ax.set_title(f'Panel{panel}: {y_label} vs. {x_label}',
                 fontfamily='Copperplate Gothic Bold', loc='left', pad=10)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend_.remove()

    return ax, handles, labels


def scatter_fig(df: pd.DataFrame,
                x: any,
                y: str,
                save: bool = False) -> None:
    """Scatter plots of n variables."""
    if not isinstance(x, list):
        x = [x]

    cols = len(x)

    col_str = '_'.join(x)

    fig, ax = plt.subplots(1, cols, figsize=(5 * cols, 4), gridspec_kw={'wspace': 0.05})

    if cols == 1:
        fig.set_size_inches(5, 4)
        scatter_plot(df, x[0], y, ax, ' a')
        handles, labels = ax.get_legend_handles_labels()

    else:
        for i in range(cols):
            scatter_plot(df, x[i], y, ax[i], f' {chr(97 + i)}')

        handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(handles, labels,
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1), frameon=False)

    plt.show()

    if save:
        fig.savefig(f'../results/figures/scatter_{y}_vs_{col_str}.png', bbox_inches='tight')


def add_constant(df: pd.DataFrame, inplace: bool = False):
    """
    Add a constant to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to add a constant to.
    inplace : bool, optional
        Whether to add the constant to the DataFrame in place.
        The default is False.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with a constant added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(inplace, bool):
        raise TypeError("inplace must be a boolean")

    if inplace:
        sm.add_constant(df, prepend=True, has_constant='skip', inplace=True)

    else:
        df = df.copy()
        df = sm.add_constant(df, prepend=True, has_constant='skip')

        return df


def hausman_test(fixed, random, save: bool = False):
    """
    This function performs the Hausman test for the fixed effects model.
    """
    # Extract the coefficients and covariance matrices
    b = fixed.params.drop('Constant', axis=0)
    B = random.params.drop('Constant', axis=0)
    V_b = fixed.cov.drop('Constant', axis=0).drop('Constant', axis=1)
    V_B = random.cov.drop('Constant', axis=0).drop('Constant', axis=1)

    # Calculate the difference in coefficients and covariance matrices
    diff = b - B
    V_diff = V_b - V_B

    # Calculate the S.E. of the difference in covariance matrices
    diff_cov_se = np.sqrt(np.diag(V_diff))

    # Calculate the chi2 statistic
    chi2 = diff.T @ np.linalg.inv(V_diff) @ diff

    # Get the degrees of freedom
    df = len(b)

    # Calculate the p-value
    pval = stats.chi2.sf(chi2, df)

    # Create a dataframe to display the results
    results = pd.DataFrame({'(b) Fixed effect': b,
                            '(B) Random effect': B,
                            '(b - B) Difference': diff,
                            'sqrt[diag(V_b - V_B)] S.E.': diff_cov_se
                            })

    results.index.name = 'Variable'

    # results.drop('const', inplace=True)

    if save:
        results.to_csv('../results/tables/hausman_test.csv', float_format="%.2f", index=True)

    # Display the results
    print('\nHausman specification test results')
    display(results.round(4))
    print('b = consistent under Ho and Ha; obtained from PanelOLS')
    print('B = inconsistent under Ha, efficient under Ho; obtained from RandomEffects')
    print('Test Ho: difference in coefficients not systematic')
    print(f'chi2({df}) = {chi2:.4f}')
    print(f'Prob > chi2 = {pval:.4f}\n')

    if pval < 0.05:
        print('The test is significant at the 5% level.')
        print('Reject Ho: difference in coefficients is systematic')
        print('Conclusion: use fixed effects model')
        print('Use fixed effects model.')
    else:
        print('The test is not significant at the 5% level.')
        print('Fail to reject Ho: difference in coefficients is not systematic')
        print('Conclusion: use random effects model')


def resid_plots(model, globals, save: bool = False):
    # Get model variable name
    model_name = [k for k, v in globals.items() if v is model][0].capitalize()

    # Create a figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'wspace': 0.05})

    # Residuals vs. fitted values
    sns.regplot(x=model.fitted_values, y=model.resids, lowess=True, ax=ax[0])
    ax[0].set_title(f'Panel a: Residuals vs. Fitted Values ({model_name})',
                    fontfamily='Copperplate Gothic Bold', loc='left', pad=10)
    ax[0].set_xlabel('Fitted Values')
    ax[0].set_ylabel('Residuals')
    ax[0].axhline(y=0, color='gray', linestyle='--')

    # QQ plot
    sm.qqplot(model.resids, line='s', ax=ax[1])
    ax[1].set_title(f'Panel b: Normal Q-Q ({model_name})',
                    fontfamily='Copperplate Gothic Bold', loc='left', pad=10)
    ax[1].set_xlabel('Theoretical Quantiles')
    ax[1].set_ylabel('Sample Quantiles')

    # Scale-location plot
    sns.regplot(x=model.fitted_values, y=np.sqrt(np.abs(model.resids)),
                lowess=True, ax=ax[2])
    ax[2].set_title(f'Panel c: Scale-Location ({model_name})',
                    fontfamily='Copperplate Gothic Bold', loc='left', pad=10)
    ax[2].set_xlabel('Fitted Values')
    ax[2].set_ylabel('Square Root of Standardized Residuals')

    # Save the figure
    if save:
        plt.savefig(f'../results/figures/resid_plots_{model_name.lower()}.png')

    # Show the figure
    plt.show()

    return None


def vif(df, save: bool = False):
    # Calculate the VIFs for the features
    vifs = pd.DataFrame()

    vifs['Features'] = df.columns
    vifs['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    vifs.set_index('Features', inplace=True)

    vifs.drop('Constant', inplace=True)

    print('Variance Inflation Factors')
    display(vifs)

    if save:
        vifs.to_csv('../results/tables/vifs.csv', float_format="%.2f", index=True)


def acf_pcaf_plots(model, globals, lags: int = 20, save: bool = False):
    # Get model name
    model_name = [k for k, v in globals.items() if v is model][0]

    # Create a figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the acf and pacf plots
    plot_acf(model.resids, lags=lags, ax=ax[0])
    plot_pacf(model.resids, lags=lags, ax=ax[1])

    # ax[0].set_title(f'Panel A: ACF of {model_name} Residuals',
    #                 fontfamily='Copperplate Gothic Bold', loc='left', pad=10)
    # ax[1].set_title(f'Panel B: PACF of {model_name} Residuals',
    #                 fontfamily='Copperplate Gothic Bold', loc='left', pad=10)

    plt.suptitle(f'{model_name.capitalize()} Residuals',
                 fontfamily='Copperplate Gothic Bold', x=0.165, y=1.1)

    if save:
        plt.savefig(f'../results/figures/acf_pacf_{model_name}.png')

    plt.show()

    return None


def plot_time_series(df: pd.DataFrame, column: str):
    """
    Plot the time series of a variable in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to show.
    column : str
        The column to show.

    Returns
    -------
    None.
    """
    fig = px.line(df, x=df.index.get_level_values("Year"), y=column,
                  color=df.index.get_level_values("Country"),
                  title=f"Time series of {column}")
    fig.update_layout(template="presentation", font_family="Times New Roman",
                      showlegend=True, legend_title_text="Country")
