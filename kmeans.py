import numpy as np


class KMeansClustering:
    def __init__(self, n_clusters=2, max_iterations=100):
        self.K = n_clusters
        self.max_iterations = max_iterations

    def init_random_centroids(self, X):
        self.num_examples, self.num_features = X.shape
        centroids = np.zeros((self.K, self.num_features))
        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid
        return centroids

    def create_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for i, x in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum((x - centroids) ** 2, axis=1)))
            clusters[closest_centroid].append(i)
        return clusters

    def calc_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids

    def predict(self, clusters, X):
        y_pred = np.zeros(self.num_examples)
        for cluster_idx, cluster in enumerate(clusters):
            for data_idx in cluster:
                y_pred[data_idx] = cluster_idx
        return y_pred

    def fit(self, X):
        centroids = self.init_random_centroids(X)
        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)
            prev_centroids = centroids
            centroids = self.calc_new_centroids(clusters, X)
            diff = centroids - prev_centroids
            if not diff.any():
                print("Converged.")
        y_pred = self.predict(clusters, X)
        return y_pred, centroids


if __name__ == "__main__":
    data = np.array([
        [1,3], [1.5,2.5], [2,2.5], [3,1.5],
        [5,7], [5.5,6.5], [6,7], [6,6.5]
    ])
    my_km = KMeansClustering(n_clusters=2, max_iterations=5)
    y_pred, clusters = my_km.fit(data)
    print(clusters)

    # test implementation
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=0).fit(data)
    print(km.cluster_centers_)
