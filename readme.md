### Dependencies

- numpy
- matplotlib
- seaborn
- prince
- statsmodels
- pandas
- pyclustertend
- sklearn

### Usage

You can run the clustering models with:

```python
python main.py --dataset <dataset> 
               --clusteringAlg <clusteringAlg>
               --max_num_clusters <max_num_clusters> 
               --num_clusters <num_clusters>
               --affinity <affinity>
               --linkage <linkage>
               --cluster_tend <cluster_tend>
```
Specifying the parameters  according to the following table:

| Parameter           | Description                                                                                                                                                                                                                                                                                                                    |
| ------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **dataset**         | Dataset to use. If set to 'vote', the vote dataset is used. If set to 'iris', the iris dataset will be used.                                                                                                                                                                                                                   |
| **clusteringAlg**    | Clustering algorithm to use. I set to 'km', the k-means algorithm is used. If set to 'bkm', the bisecting k-means algorithm will be used. If set to 'ms', the mean-shift algorithm will be used. If set to 'agg', the agglomerative clustering algorithm will be used. If set to 'kmed', the k-medoids algorithm will be used. |
| **max_num_clusters**   | If num_clusters is not specified, the max_num_cluster parameter will be used to evaluate the algorithms with different number of clusters. Results will be saved in a csv file in the results folder.                                                                                                                          |
| **num_clusters** | If the num_clusters parameter is specified, the clustering model selected will use those number of clusters and create different plots to evaluate the clustering depending on the dataset which will be saved in the plot directory.                                                                                          |
| **affinity**  | This parameter is only used if the clusteringAlg parameter is set to 'agg'. Denotes the affinity distance to use. Possible choices: ['euclidean', 'cosine'].                                                                                                                                                                   |
| **linkage**        | This parameter is only used if the clusteringAlg parameter is set to 'agg'. Denotes the kind of linkage to use. Possible choices: ['ward', 'complete', 'average', 'single'].                                                                                                                                                   |
| **cluster_tend**       | If cluster_tend parameter is set to True, the Hopkins statistic and the VAT matrix are computed. The VAT matrix will be saved in the plots folder.                                                                                                                                                                             |                |