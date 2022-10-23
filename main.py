import argparse
from datasets import preprocess_vote, preprocess_adult, preprocess_iris, preprocess_cmc, preprocess_hypothyroid
from evaluation import evaluate_clustering_number, cluster_tendency, clusterElection_plot, ari_plot, dbi_sc_ari_plot
from sklearn.cluster import AgglomerativeClustering, MeanShift
from visualize import bar_plot_vote, coordinate_plot, coordinate_plot_by_cluster, anova, chi2, confusion_matrix_compute
import pandas as pd
import matplotlib.pyplot as plt
from algorithms import FuzzyClustering, KMeans, KMedians, BKM
### Clustering Algorithms

parser = argparse.ArgumentParser()

### run--> python3 main.py --dataset vote
parser.add_argument("--dataset", type=str, default='vote', choices=['vote', 'adult', 'iris', 'cmc', 'hyp'])
parser.add_argument("--clusteringAlg", type=str, default='agg', choices=['km', 'bkm', 'ms', 'agg', 'kmed', 'fuzzy'])
parser.add_argument("--max_num_clusters", type=int, default=7, choices=range(2,100))
parser.add_argument("--num_clusters", type=int)
# For Agglomerative clustering parameters
parser.add_argument("--affinity", type=str, default = 'euclidean', choices=['euclidean', 'cosine'])
parser.add_argument("--linkage", type=str, default = 'ward', choices=['ward', 'complete', 'average', 'single'])
# Cluster tendency parameter
parser.add_argument("--cluster_tend", type=bool, default=False)
parser.add_argument("--m", type=int, default=2)
parser.add_argument("--visualize_results", type=bool, default=False)
con = parser.parse_args()

def configuration():
    config = {
                'dataset':con.dataset,
                'clusteringAlg':con.clusteringAlg,
                'max_num_clusters':con.max_num_clusters,
                'num_clusters': con.num_clusters,
                'affinity': con.affinity,
                'linkage': con.linkage,
                'cluster_tend':con.cluster_tend,
                'm':con.m,
                'visualize_results':con.visualize_results
             }
    return config

def main():
    config = configuration()
    if config['visualize_results']:
        clusterElection_plot(config)
        if config['dataset'] == 'vote':
            ari_plot(config, {'fuzzy':2,'agg_euclidean_ward':2,'agg_euclidean_complete':2, 'agg_euclidean_average':2,
                              'agg_euclidean_single':3, 'agg_cosine_complete':2, 'agg_cosine_average':2, 'agg_cosine_single':3, 'bkm':2, 'km':2, 'kmed':2, 'ms':2})
            dbi_sc_ari_plot(config, {'fuzzy':2,'agg_euclidean_ward':2,'agg_euclidean_complete':2, 'agg_euclidean_average':2,
                              'agg_euclidean_single':3, 'agg_cosine_complete':2, 'agg_cosine_average':2, 'agg_cosine_single':3, 'bkm':2, 'km':2, 'kmed':2, 'ms':2})

        if config['dataset'] == 'hyp':
            ari_plot(config, {'km': 3, 'bkm': 2, 'kmed': 3, 'agg_euclidean_ward': 3, 'fuzzy':2,
                              'agg_euclidean_complete': 2, 'agg_euclidean_average': 2, 'agg_euclidean_single': 2,
                              'agg_cosine_complete': 9, 'agg_cosine_average': 5, 'agg_cosine_single':2, 'ms':21})
            dbi_sc_ari_plot(config, {'km': 3, 'bkm': 2, 'kmed': 3, 'agg_euclidean_ward': 3, 'fuzzy':2,
                              'agg_euclidean_complete': 2, 'agg_euclidean_average': 2, 'agg_euclidean_single': 2,
                              'agg_cosine_complete': 9, 'agg_cosine_average': 5, 'agg_cosine_single':2, 'ms':21})
        if config['dataset'] == 'iris':
            ari_plot(config, {'fuzzy': 2, 'agg_euclidean_ward': 2, 'agg_euclidean_complete': 3, 'agg_euclidean_average': 2,
                              'agg_euclidean_single': 2, 'agg_cosine_complete': 2, 'agg_cosine_average': 2,
                              'agg_cosine_single': 3, 'bkm':2, 'km':2, 'kmed':2, 'ms': 2})
            dbi_sc_ari_plot(config, {'fuzzy': 2, 'agg_euclidean_ward': 2, 'agg_euclidean_complete': 3, 'agg_euclidean_average': 2,
                              'agg_euclidean_single': 2, 'agg_cosine_complete': 2, 'agg_cosine_average': 2,
                              'agg_cosine_single': 3, 'bkm':2, 'km':2, 'kmed':2, 'ms': 2})
        return

    if config['dataset'] == 'vote':
        X, Y = preprocess_vote()

    if config['dataset'] == 'adult':
        X, Y = preprocess_adult()

    if config['dataset'] == 'iris':
        X, Y = preprocess_iris()

    if config['dataset'] == 'cmc':
        X, Y = preprocess_cmc()

    if config['dataset'] == 'hyp':
        X, Y = preprocess_hypothyroid()

    if config['cluster_tend']:
        cluster_tendency(X, config)
        return

    if config['num_clusters'] is None:
        ### Evaluate different number of clusters on dataset
        result = evaluate_clustering_number(config, X, Y)
        return

    else:
        ### Run clusteringAlg on selected number of clusters
        if config['clusteringAlg'] == 'agg':
            clustering = AgglomerativeClustering(n_clusters = config['num_clusters'],
                                                 affinity=config['affinity'],
                                                 linkage=config['linkage'])
            labels = clustering.fit_predict(X.values)

        if config['clusteringAlg'] == 'ms':
            clustering = MeanShift()
            labels = clustering.fit_predict(X.values)

        if config['clusteringAlg'] == 'fuzzy':
            clustering = FuzzyClustering(n_clusters=config['num_clusters'], m=config['m'])
            labels = clustering.fit_predict(X.values)

        if config['clusteringAlg'] == 'km':
            clustering = KMeans(n_clusters=config['num_clusters'])
            labels = clustering.fit_predict(X.values)

        if config['clusteringAlg'] == 'kmed':
            clustering = KMedians(n_clusters=config['num_clusters'])
            labels = clustering.fit_predict(X.values)

        if config['clusteringAlg'] == 'bkm':
            clustering = BKM(n_clusters=config['num_clusters'])
            labels = clustering.fit_predict(X.values)

        if config['dataset'] == 'vote':
            confusion_matrix_compute(labels, Y.replace({0:'republican', 1:'democrat'}), output='./plots/vote/')
            vbles_to_perform_x2 = X.columns
            chi2(X.copy(), labels, vbles_to_perform_x2, output = './results/vote_' )
            bar_plot_vote(X.copy(), Y.copy(), labels, output = './plots/vote/', rename_target={0:'republican', 1:'democrat'})
            #correspondence_analysis_plots(X.copy(), Y.copy(), labels, output = './plots/vote/', hue='cluster', rename_target={0:'republican', 1:'democrat'})
            #correspondence_analysis_plots(X.copy(), Y.copy(), labels, output ='./plots/vote/', hue='target',   rename_target={0: 'republican', 1: 'democrat'})

        if config['dataset'] == 'hyp':
            confusion_matrix_compute(labels, Y.replace({'negative': 0, 'compensated_hypothyroid': 1, 'primary_hypothyroid': 2,
                                                        'secondary_hypothyroid': 3}), output='./plots/hyp/')

            numeric_vbles = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

            binary_vbles = ['sex', 'on_thyroxine', 'query_on_thyroxine',
                            'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                            'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                            'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured',
                            'T3_measured', 'TT4_measured', 'T4U_measured',
                            'FTI_measured']

            bar_plot_vote(X.copy(), Y.copy(), labels, output = './plots/hyp/')
            coordinate_plot_by_cluster(X.copy(), Y.copy(), labels, numeric_vbles, dataset = 'hyp', output='./plots/hyp/')
            anova(X.copy(), labels, numeric_vbles, output='./results/hyp_')
            chi2(X.copy(), labels, binary_vbles, output = './results/hyp_' )

        if config['dataset'] == 'iris':
            confusion_matrix_compute(labels, Y, output='./plots/iris/')
            columns_to_plot = [ 'petalwidth', 'petallength', 'sepalwidth', 'sepallength']
            coordinate_plot(X.copy(), Y.copy(), labels, columns_to_plot, output='./plots/iris/')
            coordinate_plot_by_cluster(X.copy(), Y.copy(), labels, columns_to_plot, output='./plots/iris/')
            anova(X.copy(), labels, columns_to_plot, output = './results/iris_')

if __name__ == '__main__':
	main()