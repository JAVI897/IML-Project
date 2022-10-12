import argparse
from datasets import preprocess_vote, preprocess_adult, preprocess_iris
from evaluation import evaluate_clustering_number, cluster_tendency
from sklearn.cluster import AgglomerativeClustering, MeanShift
from visualize import bar_plot_vote, coordinate_plot, coordinate_plot_by_cluster, anova, chi2
import pandas as pd
import matplotlib.pyplot as plt
from algorithms import FuzzyClustering
### Clustering Algorithms

parser = argparse.ArgumentParser()

### run--> python3 main.py --dataset vote
parser.add_argument("--dataset", type=str, default='vote', choices=['vote', 'adult', 'iris'])
parser.add_argument("--clusteringAlg", type=str, default='agg', choices=['km', 'bkm', 'ms', 'agg', 'kmed', 'khm', 'fuzzy'])
parser.add_argument("--max_num_clusters", type=int, default=10, choices=range(2,100))
parser.add_argument("--num_clusters", type=int)
# For Agglomerative clustering parameters
parser.add_argument("--affinity", type=str, default = 'euclidean', choices=['euclidean', 'cosine'])
parser.add_argument("--linkage", type=str, default = 'ward', choices=['ward', 'complete', 'average', 'single'])
# Cluster tendency parameter
parser.add_argument("--cluster_tend", type=bool, default=False)
parser.add_argument("--m", type=int, default=2)
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
                'm':con.m
             }
    return config

def main():
    config = configuration()

    if config['dataset'] == 'vote':
        X, Y = preprocess_vote()

    if config['dataset'] == 'adult':
        X, Y = preprocess_adult()

    if config['dataset'] == 'iris':
        X, Y = preprocess_iris()

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

        if config['dataset'] == 'vote':
            vbles_to_perform_x2 = X.columns
            chi2(X.copy(), labels, vbles_to_perform_x2, output = './results/vote_' )
            bar_plot_vote(X.copy(), Y.copy(), labels, output = './plots/vote/', rename_target={0:'republican', 1:'democrat'})
            correspondence_analysis_plots(X.copy(), Y.copy(), labels, output = './plots/vote/', hue='cluster', rename_target={0:'republican', 1:'democrat'})
            correspondence_analysis_plots(X.copy(), Y.copy(), labels, output ='./plots/vote/', hue='target',   rename_target={0: 'republican', 1: 'democrat'})

        if config['dataset'] == 'iris':
            columns_to_plot = [ 'standardscaler__petalwidth', 'standardscaler__petallength',
                                'standardscaler__sepalwidth', 'standardscaler__sepallength']
            #coordinate_plot(X.copy(), Y.copy(), labels, columns_to_plot, output='./plots/iris/')
            #coordinate_plot_by_cluster(X.copy(), Y.copy(), labels, columns_to_plot, output='./plots/iris/')
            anova(X.copy(), labels, columns_to_plot, output = './results/iris_')

if __name__ == '__main__':
	main()