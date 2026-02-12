import random
import pandas as pd
import netrd
import numpy as np
import networkx as nx
import kmapper as km

from scipy import stats
from community_detection.community import Community

def get_bootstrap_samples(df, n_bootstrap=10, ratio=0.7, seed_value=42):
    data_idx = df.index.tolist()
    idx_bootstrap = {}
    bootstrap_data = []

    random.seed(seed_value)

    for i in range(0, n_bootstrap):
        y = random.sample(data_idx, round(len(data_idx) * ratio))
        idx_bootstrap[i] = y
        idx_bootstrap_df = pd.DataFrame(idx_bootstrap)

        current_idx = idx_bootstrap_df.iloc[:, i].tolist()
        all_data = df.iloc[current_idx, :]
        bootstrap_data.append(all_data)

    X_arrays = [array.to_numpy() for array in bootstrap_data]
    return X_arrays

def graph_distance_metric(graphs_list, type: str):
    distance = []
    n_graphs = len(graphs_list)

    for i in range(0, n_graphs):
        for j in range(i + 1, n_graphs):
            graph_i = km.adapter.to_nx(graphs_list[i])
            graph_j = km.adapter.to_nx(graphs_list[j])

            if type == 'netsimile':
                dist = netrd.distance.NetSimile().dist(graph_i, graph_j)
            elif type == 'laplacian':
                dist = netrd.distance.LaplacianSpectral().dist(graph_i, graph_j)

            distance.append(dist)
    
    distance = np.array(distance)
    return np.mean(distance)

def clustering_coefficient(graphs_list):
    clustering_coeffs = []
    n_graphs = len(graphs_list)

    for i in range(0, n_graphs):
        graph_i = km.adapter.to_nx(graphs_list[i])
        clustering_coeff = nx.average_clustering(graph_i)
        clustering_coeffs.append(clustering_coeff)

    return np.mean(clustering_coeffs)


