import kmapper as km
import numpy as np

from tda.clusterers import Clusterer
from tda.filters import Filter
from scipy.spatial import distance

class TDAGraph():
    '''
    Class to create a TDA graph using KeplerMapper.
    '''
    def __init__(self, 
                 X,
                 filter_func: str,
                 n_cubes: int,
                 perc_overlap: float,
                 cluster_algorithm: str,
                 min_samples: int,
                 metric: str,
                 eps: float = None,
                 seed_value: int = 42
                 ):
        super().__init__()
        '''
        Initialize the TDAGraph object.
        
        Parameters:
        - X: Input data
        - filter_func: Name of the filter function to be applied
        - n_cubes Number of intervals (cubes) for the cover
        - perc_overlap: Percentage of overlap between intervals
        - cluster_algorithm: Name of the clustering algorithm to use
        '''
        self.X = X
        
        self.filter_func = filter_func
        if filter_func == 'tsne' and metric == 'cosine':
            raise ValueError('tsne does not support cosine metric')
        self.filter = Filter(filter_func, metric, seed_value).projector()
        self.clusterer = Clusterer(cluster_algorithm, X, min_samples, metric, eps)
        self.n_cubes = n_cubes
        self.perc_overlap = perc_overlap
        self.cluster_algorithm = cluster_algorithm

        self.mapper = km.KeplerMapper(verbose=0)

        self.filter_fitted = None

    def filtering(self):
        '''
        Apply the selected filter function to project input data.
        '''
        if self.filter_func == 'pca' and self.filter_func == 'kpca':
            X_ = distance.squareform(distance.pdist(self.X, metric=self.filter_metric))
            filtered_X = self.filter.fit_transform(X_)
        elif self.filter_func == 'tsne':
            self.filter_fitted = self.filter.fit(self.X)
            filtered_X = np.array(self.filter_fitted)
            # filtered_X = np.array(self.filter.fit(self.X))
        else:
            filtered_X = self.filter.fit_transform(self.X)
        # filtered_X = self.mapper.fit_transform(self.X, projection=self.filter)
        return filtered_X
    
    def sample_filtering(self, sample):
        if self.filter_func == 'tsne':
            filtered_sample = self.filter_fitted.transform(sample)
        else:
            filtered_sample = self.filter.transform(sample)
        return filtered_sample
    
    def covering(self):
        '''
        Define the cover (i.e., overlapping bins) over the filtered space.
        '''
        cover = km.Cover(n_cubes=self.n_cubes, perc_overlap=self.perc_overlap)
        return cover
    
    def clustering(self):
        '''
        Retrieve the selected clustering algorithm from the predefined dictionary.
        '''
        algorithm, self.eps_ = self.clusterer.cluster()
        return algorithm
    
    def create_graph(self):
        '''
        Construct the topological graph using filter, cover, and clustering.
        '''
        graph, bins, lens, cover = self.mapper.map(
                    lens=self.filtering(),
                    X=self.X,
                    cover=self.covering(),
                    clusterer=self.clustering()
                )
        
        print(f'Found optimized eps: {self.eps_}')
        return graph, bins, lens, cover