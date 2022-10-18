import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from scipy.spatial.distance import cdist

class FuzzyClustering():
    def __init__(self, n_clusters, m, epsilon = 1e-5, max_iterations=80):
        self.m = m
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.error_convergence = 0.01
        self.max_iterations = max_iterations

    def fit_predict(self, X):
        X = X
        N = X.shape[0]
        np.random.seed(34)
        #random_data_points = np.random.randint(0, high=N, size=self.n_clusters )

        # get far apart centroids
        dist_matrix = euclidean_distances(X, X)
        random_row = dist_matrix[np.random.randint(0, high=N, size=1 ),:]
        distant_points = np.argsort(random_row)[0][::-1]

        v = [X[i]+self.epsilon for i in distant_points]

        iter = 0
        errors = []
        while iter < self.max_iterations:
            print('[INFO] Iteration {}'.format(iter))

            distance_manhattan = manhattan_distances(v, X)
            u = self.update_u( v, N, distance_manhattan)
            old_v = v.copy()
            v = self.update_v(u, X, N)

            error = sum( manhattan_distances([old_v[c]], [v[c]])[0][0] for c in range(self.n_clusters))
            errors.append(error)
            print('[INFO] Error: {}'.format(error))

            if error < self.error_convergence:
                break
            iter += 1

        self.performance_index = self.compute_p(X, v, u)
        self.u = u
        self.v = v
        self.N = N
        self.errors = errors
        labels = np.argmax(self.u, axis = 0)

        return labels

    def plot_errors(self, output):
        fig = plt.figure(figsize=(20, 10))
        plt.plot(list(range(len(self.errors))), self.errors)
        plt.title('Errors Fuzzy c-means')
        plt.xlabel('Iterations')
        plt.ylabel('Termination measure $ || V_t - V_{t-1} || $')
        plt.savefig(output+'errors_c_means_c_{}'.format(self.n_clusters), bbox_inches='tight')

    def plot_u_matrix(self, output):
        plt.style.use('seaborn-white')
        fig, axs = plt.subplots(nrows=self.n_clusters, ncols=1, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.2)
        colors = ['#689F38', '#039BE5', '#FF6F00', '#C62828', '#03A9F4', '#5E35B1', '#FFCA28', '#26C6DA']
        fig.suptitle("Rows of U. Membership function", fontsize=18, y=0.95)
        for c, ax in zip(range(self.n_clusters), axs.ravel()):
            ax.plot(list(range(self.N)), self.u[c], color = colors[c] if len(colors) > self.n_clusters else '#039BE5' )
            ax.set_title('Cluster {}'.format(c))
            ax.set_ylabel('Membership')
            ax.set_ylim(0, 1)
        plt.savefig(output+'membership_c_means_c_{}'.format(self.n_clusters), bbox_inches='tight', dpi = 800)

    def compute_p(self, X, v, u):
        mean = np.mean(X, axis=0)
        p = 0
        for i_cluster, cluster in enumerate(v):
            second_sum = 0
            for i_ind, ind in enumerate(X):
                second_sum += (u[i_cluster][i_ind]**self.m) * (np.linalg.norm(ind-cluster)**2 - np.linalg.norm(cluster-mean)**2)
            p += second_sum
        return p

    def update_u(self, v, N, distance):
        power = 2 / (self.m - 0.9999999999)
        new_u = []
        for cluster in range(self.n_clusters):
            new_u_cluster = []
            for ind in range(N):
                u_cluster_ind = 1 / (np.sum([(distance[cluster, ind] / distance[cluster2, ind]) ** power for cluster2 in range(self.n_clusters)]))
                new_u_cluster.append(u_cluster_ind)
            new_u.append(new_u_cluster)
        return new_u

    def update_v(self, u, X, N):
        new_v = []
        for cluster in range(self.n_clusters):
            denom = np.sum([(u[cluster][ind] ** self.m) for ind in range(N)])
            num = np.sum([(u[cluster][ind] ** self.m) * x_i for ind, x_i in enumerate(X)], axis=0)
            new_v.append(num/denom)
        return new_v


class KMeans():
    def __init__(self, n_clusters, max_iterations=80):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit_predict(self, x):
        x = np.array(x, dtype=np.float32)
        k = self.n_clusters
        n = np.size(x, 0)
        np.random.seed(5)

        cluster_centers = x[np.random.choice(range(0, n), k, False)]
        clusters = np.empty((k, 0))
        old_clusters = []

        equal = 0
        iter = 0

        while not equal or iter < self.max_iterations:
            old_clusters = clusters

            cluster_distances = np.zeros((n, k))
            for cluster in range(k):
                cluster_distances[:, cluster] = np.sum(np.sqrt((x - cluster_centers[cluster])**2), 1)
            #print(cluster_distances)
            # "clusters" works as the indexes for the data belonging to each cluster
            clusters = np.argmin(cluster_distances, 1)
            print('-----')
            for c in range(k):
                print(x[clusters == c].shape[0])
            cluster_centers = np.array(
                [np.mean(x[clusters == c], 0) for c in range(k)])

            if old_clusters.tolist() == clusters.tolist():
                equal = 1 

            iter += 1

        return clusters


class KMedians():
    def __init__(self, n_clusters, max_iterations=80):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit_predict(self, x):
        x = np.array(x, dtype=np.float32)
        k = self.n_clusters
        n = np.size(x, 0)
        cluster_centers = x[np.random.choice(range(0, n), k, False)]
        clusters = np.empty((k, 0))
        old_clusters = []

        equal = 0
        iter = 0

        while not equal or iter < self.max_iterations:
            old_clusters = clusters

            cluster_distances = np.zeros((n, k))
            for cluster in range(k):
                cluster_distances[:, cluster] = np.sum(np.abs(x - cluster_centers[cluster]), 1)

            # "clusters" works as the indexes for the data belonging to each cluster
            clusters = np.argmin(cluster_distances, 1)
            cluster_centers = np.array(
                [np.median(x[clusters == c], 0) for c in range(k)])

            if old_clusters.tolist() == clusters.tolist():
                equal = 1 

            iter += 1

        return clusters


class BKM():
    def __init__(self, n_clusters, max_iterations=80):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit_predict(self, x):

        k = self.n_clusters
        clusters = np.zeros(len(x))
        input = np.array(x)
        n = 0
        max_c = 0
        num_c = 1
        
        clustering = KMeans(n_clusters=2)

        while num_c < k:
            kmeans = clustering.fit_predict(input)
            kmeans += n #we increment by 2 the cluster indexes on each iteration to avoid cluster merging

            if k == 2: return kmeans

            #rewrite the "clusters" array to subsitute the processed cluster by its 2 bisections
            for c in range(len(clusters)):
                if clusters[c] == max_c:
                    clusters[c], kmeans = kmeans[0], kmeans[1:]

            #find the cluster with the highest number of elements
            unique, counts = np.unique(clusters, return_counts=True)
            d = dict(zip(unique, counts))
            max_c = max(d, key=d.get)

            #set as input for the next iteration the cluster with the maximum number of elements
            input = x[clusters == max_c]      

            n += 2
            num_c += 1
        
        #change the cluster numbers starting from zero to k
        i = 0
        for c in np.unique(clusters):
            clusters[np.argwhere(clusters == (c).flatten().tolist())] = i
            i += 1

        clusters = np.array(clusters, dtype=int)
        return clusters