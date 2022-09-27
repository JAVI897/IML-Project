import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score

def evaluate_clustering_number(config, X, Y):
    eval = {}
    for n in range(2, config['max_num_clusters']):
        if config['clusteringAlg'] == 'agg':
            clustering = AgglomerativeClustering(n_clusters = n, affinity=config['affinity'], linkage=config['linkage'])
        labels = clustering.fit_predict(X.values)
        ari = adjusted_rand_score(Y, labels)
        eval[n] = {'ari':ari}

    make_plots(config, eval)
    return eval

def make_plots(config, eval):
    if config['clusteringAlg'] == 'agg':
        name = 'Dataset-{}--AgglomerativeClustering-{}-{}.png'.format(config['dataset'], config['affinity'], config['linkage'])
        title = 'Evaluation for AgglomerativeClustering - Dataset: {} - Affinity: {} - Linkage: {}'.format(config['dataset'], config['affinity'], config['linkage'])

    fig = plt.figure(figsize = (20, 10))
    plt.plot(eval.keys(), [d['ari'] for d in eval.values()], label = 'Adjusted Rand Index')
    plt.title(title)
    plt.legend()
    plt.savefig('./plots/{}/{}'.format(config['dataset'], name), bbox_inches='tight')