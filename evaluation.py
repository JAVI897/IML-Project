import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
from pyclustertend import hopkins, vat
from algorithms import FuzzyClustering

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
        ari = adjusted_rand_score(Y, labels)
        sil = silhouette_score(X.values, labels)
        dbs = davies_bouldin_score(X.values, labels)
        ch  = calinski_harabasz_score(X.values, labels)
        n_clusters = len(np.unique(labels))
        eval[n_clusters] = {'ari':round(ari, 3), 'sil': round(sil, 3), 'dbs': round(dbs, 3), 'ch': round(ch, 3)}
        save_log(config, eval)
        return eval

    p_indexes = {}
    for n in range(2, config['max_num_clusters']):
        if config['clusteringAlg'] == 'agg':
            clustering = AgglomerativeClustering(n_clusters = n, affinity=config['affinity'], linkage=config['linkage'])
        if config['clusteringAlg'] == 'fuzzy':
            clustering = FuzzyClustering(n_clusters = n, m = config['m'])

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

    #make_plots(config, eval)
    save_log(config, eval)
    return eval

def plot_p_indexes(config, p_indexes, output):
    title = 'Dataset: {} Performance indexes for c-means fuzzy clustering with m={}'.format(config['dataset'], config['m'])
    print(p_indexes)
    fig = plt.figure(figsize=(13, 10))
    plt.plot(p_indexes.keys(), p_indexes.values(), label='Performance indexes')
    plt.title(title)
    plt.xlabel('Clusters')
    plt.ylabel('Performance index')
    plt.savefig(output, bbox_inches='tight')

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


def make_plots(config, eval):
    if config['clusteringAlg'] == 'agg':
        name = 'Dataset-{}--AgglomerativeClustering-{}-{}.png'.format(config['dataset'], config['affinity'], config['linkage'])
        title = 'Evaluation for AgglomerativeClustering - Dataset: {} - Affinity: {} - Linkage: {}'.format(config['dataset'], config['affinity'], config['linkage'])

    fig = plt.figure(figsize = (20, 10))
    plt.plot(eval.keys(), [d['ari'] for d in eval.values()], label = 'Adjusted Rand Index')
    plt.plot(eval.keys(), [d['sil'] for d in eval.values()], label='Silhouette Score')
    plt.title(title)
    plt.legend()
    plt.savefig('./plots/{}/{}'.format(config['dataset'], name), bbox_inches='tight')