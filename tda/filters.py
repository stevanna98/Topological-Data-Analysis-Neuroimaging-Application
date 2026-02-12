from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from openTSNE import TSNE
from umap.umap_ import UMAP

class Filter:
    def __init__(self, 
                 filter_func: str,
                 filter_metric: str = 'euclidean',
                 seed_value: int = 42):
        self.filter_func = filter_func
        self.filter_metric = filter_metric
        self.seed_value = seed_value

    def projector(self):
        if self.filter_func == 'pca':
            return PCA(n_components=2, random_state=self.seed_value)
        
        elif self.filter_func == 'kpca':
            return KernelPCA(n_components=2, kernel='rbf', random_state=self.seed_value)
        
        elif self.filter_func == 'isomap':
            return Isomap(n_components=2, metric=self.filter_metric)
        
        elif self.filter_func == 'umap':
            return UMAP(n_components=2, n_neighbors=25, random_state=self.seed_value, metric=self.filter_metric)
        
        elif self.filter_func == 'tsne':
            return TSNE(n_components=2, random_state=self.seed_value, metric=self.filter_metric)
        


