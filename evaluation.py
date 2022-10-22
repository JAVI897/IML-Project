import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
from pyclustertend import hopkins, vat
from algorithms import FuzzyClustering, KMeans, KMedians, BKM
import seaborn as sns
import math
##### CLUSTER TENDENCY FUNCTIONS

def cluster_tendency(X, config):
    """
    cluster_tendency
    Function that takes a dataset and computes
    the Hopkins statistic and the VAT matrix
    implemented using pyclustertend library

    :param X: nxd dataset
    :param config: config dictionary
    """
    output = './plots/{}/'.format(config['dataset'])
    H = hopkins(X.values, X.values.shape[0])
    print('[INFO] Hopkins test: {}'.format(H))
    vat_matrix = vat(X.values)
    plt.title('Vat matrix for dataset: {}'.format(config['dataset']))
    plt.savefig(output+'vat_matrix_{}.jpg'.format(config['dataset']), bbox_inches='tight')

##### FUNCTIONS TO EVALUATE THE NUMBER OF CLUSTERS

def evaluate_clustering_number(config, X, Y):
    """
    evaluate_clustering_number
    Function that depending on the Clustering algorithm
    that has been selected computes different clustering
    metrics for different number of clusters. Results are
    saved in a csv file

    :param config: config dictionary
    :param X: nxd dataset
    :param Y: target
    :return eval dictionary which keys are the number of
    clusters and values another dictionary with the metrics
    for each number of cluster
    """
    eval = {}

    if config['clusteringAlg'] == 'ms':
        clustering = MeanShift()
        labels = clustering.fit_predict(X.values)
        n_labels = len(np.unique(labels))
        if n_labels > 1: # if meanshift founds just 1 cluster, evaluation measures do not make sense
            ari = adjusted_rand_score(Y, labels)
            sil = silhouette_score(X.values, labels)
            dbs = davies_bouldin_score(X.values, labels)
            ch  = calinski_harabasz_score(X.values, labels)
            n_clusters = len(np.unique(labels))
            eval[n_clusters] = {'ari':round(ari, 3), 'sil': round(sil, 3), 'dbs': round(dbs, 3), 'ch': round(ch, 3)}
        else:
            eval[n_labels] = {'ari': None, 'sil': None, 'dbs': None, 'ch': None}
        save_log(config, eval)
        return eval

    p_indexes = {}
    for n in range(2, config['max_num_clusters']):
        print('[INFO] Testing for k = {} clusters'.format(n))
        if config['clusteringAlg'] == 'agg':
            clustering = AgglomerativeClustering(n_clusters = n, affinity=config['affinity'], linkage=config['linkage'])
        if config['clusteringAlg'] == 'fuzzy':
            clustering = FuzzyClustering(n_clusters = n, m = config['m'])
        if config['clusteringAlg'] == 'km':
            clustering = KMeans(n_clusters=n)
        if config['clusteringAlg'] == 'kmed':
            clustering = KMedians(n_clusters=n)
        if config['clusteringAlg'] == 'bkm':
            clustering = BKM(n_clusters=n)

        labels = clustering.fit_predict(X.values)
        ari = adjusted_rand_score(Y, labels)
        sil = silhouette_score(X.values, labels)
        dbs = davies_bouldin_score(X.values, labels)
        ch = calinski_harabasz_score(X.values, labels)
        eval[n] = {'ari':round(ari, 3), 'sil': round(sil, 3), 'dbs': round(dbs, 3), 'ch': round(ch, 3)}

        # save other plots for fuzzy clustering
        if config['clusteringAlg'] == 'fuzzy':
            p_indexes[n] = clustering.performance_index
            clustering.plot_errors('./plots/{}/c_means/'.format(config['dataset']))
            clustering.plot_u_matrix('./plots/{}/c_means/'.format(config['dataset']))

    if config['clusteringAlg'] == 'fuzzy':
        plot_p_indexes(config, p_indexes, output='./plots/{}/c_means/p_indexes.png'.format(config['dataset']))
        plot_p_indexes_and_others(config, p_indexes, eval, output='./plots/{}/c_means/p_indexes_others.png'.format(config['dataset']))

    #make_plots(config, eval)
    save_log(config, eval)
    return eval


def save_log(config, eval):
    """
    save_log
    Function that takes the eval dictionary and saves the results
    in an existing csv file that gets updated with the new results

    :param config: config dictionary
    :param eval: eval dictionary
    """
    path = './results/{}.csv'.format(config['dataset'])
    data = [[k, v['ari'], v['sil'], v['dbs'], v['ch'] ] for k, v in eval.items()]
    if os.path.isfile(path):
        df = pd.read_csv(path)
        df_aux = pd.DataFrame(data, columns = ['Number of clusters', 'ari', 'sil', 'dbs', 'ch'])
        df_aux['dataset'] = config['dataset']
        df_aux['clusteringAlg'] = config['clusteringAlg']
        df_aux['affinity'] = config['affinity'] if config['clusteringAlg'] == 'agg' else 'None'
        df_aux['linkage'] = config['linkage'] if config['clusteringAlg'] == 'agg' else 'None'
        df_both = pd.concat([df, df_aux], ignore_index=True, sort=False)
        df_both = df_both.drop_duplicates()
        df_both.to_csv(path, index=False)
    else:
        df = pd.DataFrame(data, columns = ['Number of clusters', 'ari', 'sil', 'dbs', 'ch'])
        df['dataset'] = config['dataset']
        df['clusteringAlg'] = config['clusteringAlg']
        df['affinity'] = config['affinity'] if config['clusteringAlg'] == 'agg' else 'None'
        df['linkage'] = config['linkage'] if config['clusteringAlg'] == 'agg' else 'None'
        df.to_csv(path, index=False)

#### PLOT FUNCTIONS
def plot_p_indexes(config, p_indexes, output):
    title = 'Dataset: {} Performance indexes for c-means fuzzy clustering with m={}'.format(config['dataset'], config['m'])
    print(p_indexes)
    fig = plt.figure(figsize=(13, 10))
    plt.plot(p_indexes.keys(), p_indexes.values(), label='Performance indexes')
    plt.title(title)
    plt.xlabel('Clusters')
    plt.ylabel('Performance index')
    plt.savefig(output, bbox_inches='tight')

def plot_p_indexes_and_others(config, p_indexes, eval, output):
    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=(25, 8))
    colors = ['#689F38', '#039BE5', '#FF6F00', '#F44336', '#26C6DA']
    plt.subplot(1, 3, 1)
    plt.plot(eval.keys(), [d['ch'] for d in eval.values()], linestyle = 'solid', marker = 'o', color = colors[3],label='Calinski Harabasz index')
    plt.legend()
    plt.grid()
    plt.title('Evaluation for Fuzzy Clustering: CH')
    plt.xlabel('Clusters')
    plt.ylabel('CH')

    plt.subplot(1, 3, 2)
    plt.plot(eval.keys(), [d['sil'] for d in eval.values()], linestyle = 'solid', marker = 'o', color = colors[1], label='Silhouette Score')
    plt.plot(eval.keys(), [d['dbs'] for d in eval.values()], linestyle = 'solid', marker = 'o', color = colors[2], label='Davies-Bouldin index')
    plt.legend()
    plt.grid()
    plt.title('Evaluation for Fuzzy Clustering: SC and DBI')
    plt.xlabel('Clusters')
    plt.ylabel('SC and DBI')

    plt.subplot(1, 3, 3)
    plt.plot(p_indexes.keys(), p_indexes.values(), linestyle = 'solid', marker = 'o', color = colors[0], label='Performance indexes')
    plt.grid()
    plt.title('Performance indexes. C-means fuzzy clustering with m={}'.format( config['m']))
    plt.xlabel('Clusters')
    plt.ylabel('Performance index')

    plt.savefig(output, bbox_inches='tight', dpi = 500)


def clusterElection_plot(config):
    """
    Given the name of the dataset,
    this function plots a multiple line plot for each algorithm result.

    :param config: config dict
    :param output: results path
    """
    path_dataset = {"vote": './results/vote.csv', "hyp": './results/hyp.csv', "iris": './results/iris.csv'}
    X = pd.read_csv(path_dataset[config['dataset']])

    X["plot_name"] = X["clusteringAlg"] + "_" + X["affinity"] + "_" + X["linkage"]
    X["plot_name"] = X["plot_name"].apply(lambda x: x.replace('_None_None', ''))
    X = X[X["clusteringAlg"] != "ms"]
    X = X[X["clusteringAlg"] != "fuzzy"]
    X_melt = pd.melt(X, id_vars=["plot_name", "Number of clusters"], value_vars=["sil", "dbs"], var_name="metrics")

    #print(min(X_melt["metrics"]))
    alg_list = X_melt["plot_name"].unique()

    #plt.style.use('seaborn-white')
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(math.ceil(len(alg_list) / 4), 4, figsize=(25, 15))
    for n, alg in enumerate(alg_list):
        x_subset = X_melt[X_melt["plot_name"] == alg]

        sns.lineplot(data=x_subset, x="Number of clusters", y="value", hue="metrics", style="metrics", markers=['o', 'o'], dashes=False,
                     ax=ax[n // 4, n - (n // 4) * 4]).set(title=alg, ylabel='Evaluation for Fuzzy Clustering: SC and DBI',
                                                          ylim=(min(X_melt["value"]+0.1), max(X_melt["value"]+0.1)))

    output = './plots/{}/clusterElection_sil_dbs.jpg'.format(config['dataset'])
    plt.savefig(output, bbox_inches='tight')

    ## calinski_harabasz_score
    X = pd.read_csv(path_dataset[config['dataset']])

    X["plot_name"] = X["clusteringAlg"] + "_" + X["affinity"] + "_" + X["linkage"]
    X["plot_name"] = X["plot_name"].apply(lambda x: x.replace('_None_None', ''))
    X = X[X["clusteringAlg"] != "ms"]
    X_melt = pd.melt(X, id_vars=["plot_name", "Number of clusters"], value_vars=["ch"], var_name="metrics")

    #print(min(X_melt["metrics"]))
    alg_list = X_melt["plot_name"].unique()

    #plt.style.use('seaborn-white')
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(math.ceil(len(alg_list) / 4), 4, figsize=(25, 15))
    for n, alg in enumerate(alg_list):
        x_subset = X_melt[X_melt["plot_name"] == alg]

        sns.lineplot(data=x_subset, x="Number of clusters", y="value", hue="metrics", style="metrics", markers=['o'], dashes=False,
                     ax=ax[n // 4, n - (n // 4) * 4]).set(title=alg, ylabel='Evaluation for Fuzzy Clustering: CH',
                                                          ylim=(min(X_melt["value"] + 0.1), max(X_melt["value"] + 0.1)))

    output = './plots/{}/clusterElection_ch.jpg'.format(config['dataset'])
    plt.savefig(output, bbox_inches='tight')

    X = pd.read_csv(path_dataset[config['dataset']])

    X["plot_name"] = X["clusteringAlg"] + "_" + X["affinity"] + "_" + X["linkage"]
    X["plot_name"] = X["plot_name"].apply(lambda x: x.replace('_None_None', ''))
    X = X[X["clusteringAlg"] != "ms"]
    X = X[X["clusteringAlg"] != "fuzzy"]
    X_melt_sil_dbs = pd.melt(X, id_vars=["plot_name", "Number of clusters"], value_vars=["sil", "dbs"], var_name="metrics")
    X_melt_ch = pd.melt(X, id_vars=["plot_name", "Number of clusters"], value_vars=["ch"], var_name="metrics")
    #print(min(X_melt["metrics"]))
    alg_list = X_melt_sil_dbs["plot_name"].unique()

    #plt.style.use('seaborn-white')
    #sns.set_style("whitegrid")
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(math.ceil(len(alg_list) / 3), 3, figsize=(32, 35))
    for n, alg in enumerate(alg_list):
        x_subset_sil_dbs = X_melt_sil_dbs[X_melt_sil_dbs["plot_name"] == alg]
        x_subset_ch = X_melt_ch[X_melt_ch["plot_name"] == alg]

        g = sns.lineplot(data=x_subset_sil_dbs, x="Number of clusters", y="value", hue="metrics", style="metrics", markers=['o', 'o'], dashes=False,
                     ax=ax[n // 3, n - (n // 3) * 3], legend = False).set(title=alg, ylabel='SC and DBI')
        sns.lineplot(data=x_subset_ch, x="Number of clusters", y="value", color='r', marker='o', dashes=False,
                     ax=ax[n // 3, n - (n // 3) * 3].twinx()).set(title=alg, ylabel='CH')
        ax[n // 3, n - (n // 3) * 3].legend(handles=[Line2D([], [], marker='o', color="orange", label='DBI'),
                                                     Line2D([], [], marker='o', color="dodgerblue", label='SC'),
                                                     Line2D([], [], marker='o', color="r", label='CH')],
                                            loc='upper right')
        ax[n // 3, n - (n // 3) * 3].grid(True)

    ax[3,2].set_axis_off()
    ax[3,1].set_axis_off()
    output = './plots/{}/clusterElection_sil_dbs_ch.jpg'.format(config['dataset'])
    plt.savefig(output, bbox_inches='tight', dpi = 500)


def ari_plot(config, n_clust_alg_dict):
    """
    Given the name of the dataset,
    this function plots a multiple line plot for each algorithm result.

    :param config: config dict
    :param n_clust_alg_dict: number of clusters selected per algorithm
    """
    path_dataset = {"vote": './results/vote.csv', "hyp": './results/hyp.csv', "iris": './results/iris.csv'}
    X = pd.read_csv(path_dataset[config['dataset']])

    X["algorithm"] = X["clusteringAlg"] + "_" + X["affinity"] + "_" + X["linkage"] + "_" + X['Number of clusters'].astype(str)
    X["algorithm"] = X["algorithm"].apply(lambda x: x.replace('_None_None', ''))

    colors = ['#689F38', '#039BE5', '#FF6F00', '#F44336', '#26C6DA', '#FFC107', '#E91E63']
    filter_algs = ['{}_{}'.format(k, v) for k, v in n_clust_alg_dict.items()]
    X = X.loc[X['algorithm'].isin(filter_algs)]
    X_melt = pd.melt(X, id_vars=["algorithm"], value_vars=["ari"], var_name="metrics")
    X_melt = X_melt.sort_values('value')
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(data=X_melt, x="algorithm", y="value", orient='v', palette = 'tab10').set(title="ARI results")
    ax.bar_label(ax.containers[0], padding=3)
    ax.set_ylabel('ARI')
    ax.set_xlabel('')
    ax.set_xticklabels(X_melt.algorithm, rotation=45, ha='right')

    output = './plots/{}/ari.jpg'.format(config['dataset'])
    plt.savefig(output, bbox_inches='tight')


def dbi_sc_ari_plot(config, n_clust_alg_dict):
    """
    Given the name of the dataset,
    this function plots a multiple line plot for each algorithm result.

    :param config: config dict
    :param n_clust_alg_dict: number of clusters selected per algorithm
    """
    path_dataset = {"vote": './results/vote.csv', "hyp": './results/hyp.csv', "iris": './results/iris.csv'}
    X = pd.read_csv(path_dataset[config['dataset']])

    X["algorithm"] = X["clusteringAlg"] + "_" + X["affinity"] + "_" + X["linkage"] + "_" + X['Number of clusters'].astype(str)
    X["algorithm"] = X["algorithm"].apply(lambda x: x.replace('_None_None', ''))

    colors = ['#689F38', '#039BE5', '#FF6F00', '#F44336', '#26C6DA', '#FFC107', '#E91E63']
    filter_algs = ['{}_{}'.format(k, v) for k, v in n_clust_alg_dict.items()]
    X = X.loc[X['algorithm'].isin(filter_algs)]
    X_melt = pd.melt(X, id_vars=["algorithm"], value_vars=["ari", "sil", "dbs"], var_name="metrics")
    X_melt = X_melt.sort_values('value')

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(data=X_melt, x="algorithm", y="value", hue = "metrics", orient='v', palette = 'tab10', hue_order = ['dbs', 'sil', 'ari']).set(title="ARI, DBI and SC results")
    ax.bar_label(ax.containers[0], padding=3)
    ax.bar_label(ax.containers[1], padding=3)
    ax.bar_label(ax.containers[2], padding=3)
    ax.set_ylabel('ARI, DBI and SC')
    ax.set_xlabel('')
    plt.xticks(rotation=45)

    output = './plots/{}/dbi_sc_ari.jpg'.format(config['dataset'])
    plt.savefig(output, bbox_inches='tight')