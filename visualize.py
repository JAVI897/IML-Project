import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
#import prince
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd

##### STATISTICAL TEST FUNCTIONS
def chi2(X, labels, vbles_to_perform_x2, output):
    """
    chi2 test
    Function that given a dataset, a set of clustering labels
    and a list of variables computes the chi2 test for each
    categorical variable using the clustering labels. p-value
    results are saved in a csv file

    :param X: nxd dataset
    :param labels: clustering labels
    :param vbles_to_perform_x2: list of variables
    :param output: path to save csv
    """
    dict = {'Vble': [], 'chi2_p_value': []}
    for vble_cat in vbles_to_perform_x2:
        df = pd.DataFrame({'value': X[vble_cat].values, 'clusters':['cluster {}'.format(i) for i in labels]})
        table = sm.stats.Table.from_data(df)
        rslt = table.test_nominal_association()
        dict['chi2_p_value'].append( round(rslt.pvalue, 4) )
        dict['Vble'].append(vble_cat)
    x2_results = pd.DataFrame(dict)
    x2_results.to_csv(output+'x2.csv', index=False)

def anova(X, labels, vbles_to_perform_anova, output):
    """
    anova
    Function that given a dataset, a set of clustering labels
    and a list of variables computes the ANOVA test for each
    variable using as factors the clustering labels. p-value
    results are saved in a csv file

    :param X: nxd dataset
    :param labels: clustering labels
    :param vbles_to_perform_anova: list of variables
    :param output: path to save csv
    """
    dict = {'Vble': [], 'Anova_p_value': []}
    for vble_num in vbles_to_perform_anova:
        df = pd.DataFrame({'value': X[vble_num].values, 'clusters':['cluster {}'.format(i) for i in labels]})
        df["value"] = pd.to_numeric(df["value"])
        model = ols('value ~ clusters', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        dict['Anova_p_value'].append( round(anova_table['PR(>F)'].clusters, 4) )
        dict['Vble'].append(vble_num)
    anovas_results = pd.DataFrame(dict)
    anovas_results.to_csv(output+'anova.csv', index=False)

##### VISUALIZATION FUNCTIONS

def bar_plot_vote(X, Y, labels, output, rename_target = None):
    """
    bar_plot_vote
    Given a dataset, the target variable and the clustering labels,
    this function plots a bar plot for each categorical variable
    in the vote dataset distinguishing between clusters

    :param X: nxd dataset
    :param Y: target variable
    :param labels: clustering labels
    :param output: path
    """
    plt.style.use('seaborn-white')
    X['Target'] = Y
    if rename_target is not None:
        X['Target'] = X['Target'].replace(rename_target)
    X['Cluster'] = labels
    for column in X.columns:
        print(column)
        fig = plt.figure(figsize=(20, 10))
        sns.countplot(data = X, x = 'Cluster', hue=column)
        plt.savefig(output+'{}.jpg'.format(column), bbox_inches='tight')

def coordinate_plot(X, Y, labels, columns_to_plot, output, rename_target=None):
    """
    coordinate_plot
    Given a dataset, the target variable and the clustering labels,
    this function plots a coordinate plot for each variable specified
    in columns_to_plot list

    :param X: nxd dataset
    :param Y: target
    :param labels: clustering labels
    :param columns_to_plot: list of columns to plot
    :param output: output path
    """
    plt.style.use('seaborn-white')
    X['Target'] = Y
    if rename_target is not None:
        X['Target'] = X['Target'].replace(rename_target)
    X['Cluster'] = labels
    fig = plt.figure(figsize=(20, 10))
    parallel_coordinates(X, 'Target', cols = ['Cluster'] + columns_to_plot)
    plt.savefig(output + 'parallel_coords.jpg', bbox_inches='tight')


def coordinate_plot_by_cluster(X, Y, labels, columns_to_plot, output, rename_target=None):
    """
    coordinate_plot
    Given a dataset, the target variable and the clustering labels,
    this function plots a coordinate plot for each variable specified
    in columns_to_plot list and for each cluster

    :param X: nxd dataset
    :param Y: target
    :param labels: clustering labels
    :param columns_to_plot: list of columns to plot
    :param output: output path
    """
    plt.style.use('seaborn-white')
    X['Target'] = Y
    if rename_target is not None:
        X['Target'] = X['Target'].replace(rename_target)
    X['Cluster'] = labels
    max_value = max(max(X[columns_to_plot[i]]) for i in range(len(columns_to_plot)))
    min_value = min(min(X[columns_to_plot[i]]) for i in range(len(columns_to_plot)))
    for cluster in X.Cluster.unique():
        X_c = X.loc[X['Cluster'] == cluster]
        fig = plt.figure(figsize=(20, 10))
        parallel_coordinates(X_c, 'Target', cols = columns_to_plot)
        plt.title('Cluster: {}'.format(cluster))
        plt.ylim(max_value, min_value)
        plt.savefig(output + 'parallel_coords_cluster_{}.jpg'.format(cluster), bbox_inches='tight')

"""
def correspondence_analysis_plots(X, Y, labels, output, hue = 'cluster', rename_target=None):
    
    #correspondence_analysis_plots
    #Function that given a dataset, the target variable and the clustering
    #labels, performs a Multiple correspondence analysis on X colouring
    #the plots by the variable specified in the hue parameter

    #:param X: nxd dataset
    #:param Y: target
    #:param labels: clustering labels
    #:param output: path
    #:param hue: colouring
    
    plt.style.use('seaborn-white')
    X.columns = [i.replace('_', '-').replace('-yes', '') for i in X.columns]
    mca = prince.MCA()
    mca = mca.fit(X)
    if hue == 'cluster':
        clusters = labels.astype(str)
        N = len(set(clusters))
    else:
        clusters = list(Y.replace(rename_target).values)
        N = len(set(clusters))
    row_coords = mca.row_coordinates(X)
    col_coords = mca.column_coordinates(X)
    one_hot_enc = mca.enc.fit(X)
    feature_names = one_hot_enc.get_feature_names_out()
    col_coords['features'] = feature_names
    col_coords = col_coords.set_index('features')

    fig, ax = plt.subplots(figsize=(20, 10))
    row_coords['groups'] = clusters
    for group, group_row_coords in row_coords.groupby('groups'):
        ax.scatter(
            group_row_coords.iloc[:, 0],
            group_row_coords.iloc[:, 1],
            s=10,
            label=group,
            alpha = 0.8
        )

    x = col_coords[0]
    y = col_coords[1]

    prefixes = col_coords.index.str.split('_').map(lambda x: x[0])
    for prefix in prefixes.unique():
        mask = prefixes == prefix
        ax.scatter(x[mask], y[mask], s=10, marker='X', label=prefix)
        for i, label in enumerate(col_coords[mask].index):
            ax.annotate(label, (x[mask][i], y[mask][i]))

    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:N], l[:N])
    ax.set_title('Row and column principal coordinates colored by {}'.format(hue))
    ei = mca.explained_inertia_
    ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(0, 100 * ei[0]))
    ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(1, 100 * ei[1]))

    ax.get_figure().savefig(output + 'coordinates_{}.jpg'.format(hue))
    plt.close()
"""



