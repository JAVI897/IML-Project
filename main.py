import argparse
from datasets import preprocess_vote, preprocess_adult, preprocess_iris
from evaluation import evaluate_clustering_number
### Clustering Algorithms

parser = argparse.ArgumentParser()

### run--> python3 main.py --dataset vote
parser.add_argument("--dataset", type=str, default='vote', choices=['vote', 'adult', 'iris'])
parser.add_argument("--clusteringAlg", type=str, default='agg', choices=['km', 'bkm', 'ms', 'agg', 'kmed', 'khm'])
parser.add_argument("--max_num_clusters", type=int, default=10, choices=range(2,100))
parser.add_argument("--num_clusters", type=int)
# For Agglomerative clustering parameters
parser.add_argument("--affinity", type=str, default = 'euclidean', choices=['euclidean', 'cosine'])
parser.add_argument("--linkage", type=str, default = 'ward', choices=['ward', 'complete', 'average', 'single'])
con = parser.parse_args()

def configuration():
    config = {
                'dataset':con.dataset,
                'clusteringAlg':con.clusteringAlg,
                'max_num_clusters':con.max_num_clusters,
                'num_clusters': con.num_clusters,
                'affinity': con.affinity,
                'linkage': con.linkage
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


    if config['num_clusters'] is None:
        ### Evaluate different number of clusters on dataset
        result = evaluate_clustering_number(config, X, Y)
        print(result)

    else:
        ### Run clusteringAlg on selected number of clusters
        if config['clusteringAlg'] == 'agg':
            clustering = AgglomerativeClustering(n_clusters = config['num_clusters'],
                                                 affinity=config['affinity'],
                                                 linkage=config['linkage'])

if __name__ == '__main__':
	main()