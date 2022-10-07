import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import pandas as pd
import numpy as np
from pyclustertend import hopkins, vat

def cluster_tendency(X, config):
    output = './plots/{}/'.format(config['dataset'])
    H = hopkins(X.values, X.values.shape[0])
    print('[INFO] Hopkins test: {}'.format(H))
    vat_matrix = vat(X.values)
    plt.title('Vat matrix for dataset: {}'.format(config['dataset']))
    plt.savefig(output+'vat_matrix_{}.jpg'.format(config['dataset']), bbox_inches='tight')


def evaluate_clustering_number(config, X, Y):
    eval = {}
    if config['clusteringAlg'] == 'ms':
        clustering = MeanShift()
        labels = clustering.fit_predict(X.values)
        ari = adjusted_rand_score(Y, labels)
        sil = silhouette_score(X.values, labels)
        dbs = davies_bouldin_score(X.values, labels)
        n_clusters = len(np.unique(labels))
        eval[n_clusters] = {'ari':round(ari, 3), 'sil': round(sil, 3), 'dbs': round(dbs, 3)}
        save_log(config, eval)
        return eval

    for n in range(2, config['max_num_clusters']):
        if config['clusteringAlg'] == 'agg':
            clustering = AgglomerativeClustering(n_clusters = n, affinity=config['affinity'], linkage=config['linkage'])
        labels = clustering.fit_predict(X.values)
        ari = adjusted_rand_score(Y, labels)
        sil = silhouette_score(X.values, labels)
        dbs = davies_bouldin_score(X.values, labels)
        eval[n] = {'ari':round(ari, 3), 'sil': round(sil, 3), 'dbs': round(dbs, 3)}

    #make_plots(config, eval)
    save_log(config, eval)
    return eval

def save_log(config, eval):
    path = './results/{}.csv'.format(config['dataset'])
    data = [[k, v['ari'], v['sil'], v['dbs']] for k, v in eval.items()]
    if os.path.isfile(path):
        df = pd.read_csv(path)
        df_aux = pd.DataFrame(data, columns = ['Number of clusters', 'ari', 'sil', 'dbs'])
        df_aux['dataset'] = config['dataset']
        df_aux['clusteringAlg'] = config['clusteringAlg']
        df_aux['affinity'] = config['affinity'] if config['clusteringAlg'] == 'agg' else 'None'
        df_aux['linkage'] = config['linkage'] if config['clusteringAlg'] == 'agg' else 'None'
        df_both = pd.concat([df, df_aux], ignore_index=True, sort=False)
        df_both = df_both.drop_duplicates()
        df_both.to_csv(path, index=False)
    else:
        df = pd.DataFrame(data, columns = ['Number of clusters', 'ari', 'sil', 'dbs'])
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
    plt.savefig('./plots/{}/{}'.format(conig['dataset'], name), bbox_inches='tight')