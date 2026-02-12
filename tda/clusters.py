from sklearn.cluster import DBSCAN, HDBSCAN
import numpy as np

class Clusterer:
    def __init__(self,
                 cluster_algorithm: str = 'dbscan',
                 X: np.ndarray = None,
                 min_samples: int = 3,
                 distance_metric: str = 'chebyshev',
                 eps: float = 128):
        self.cluster_algorithm = cluster_algorithm
        self.X = X
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.eps = eps

    def optimize_dbscan(self, X, min_samples, k=3, p=100.0, **kwargs):
        eps = self.optimize_eps(X, k=k, p=p)
        dbscan = DBSCAN(
            eps=eps, min_samples=min_samples,
            metric=self.distance_metric, p=2, leaf_size=15,
            **kwargs
        )
        if self.eps is None:
            return dbscan, eps
        else:
            return dbscan, self.eps
    
    def optimize_eps(self, X, k=3, p=100.0, **kwargs):
        from sklearn.neighbors import KDTree

        tree = KDTree(X, metric=self.distance_metric, leaf_size=15)
        dist, ind = tree.query(X, k=k+1)
        eps = np.percentile(dist[:, k], p)
        return eps

    def cluster(self):
        if self.cluster_algorithm == 'dbscan':
            # return DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.distance_metric)
            return self.optimize_dbscan(self.X, self.min_samples)