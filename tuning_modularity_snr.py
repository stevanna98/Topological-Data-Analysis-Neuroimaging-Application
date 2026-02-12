import pandas as pd
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
from tda.tda_graph import TDAGraph
from tuning.heatmap_creation import Heatmap
from tuning.tuning_utils import *
from community_detection.community import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set random seed for reproducibility
seed_value = 42

def main(args):         
    # Load data
    fmri_df = pd.read_csv(args.fmri)
    env_df = pd.read_csv(args.env)
    demo_df = pd.read_csv(args.demo_vars)

    new_demo_df = demo_df[['SEX', 'AGE', 'EDU']]

    feature_set = ['all', 'fmri', 'env']
    filtering = ['umap', 'pca', 'tsne', 'isomap']

    # Cover parameters tuning
    perc_overlap = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_cubes = [10, 20, 30, 40, 50, 60]

    demographic = True
    for filter_ in filtering:
        for feature in feature_set:
            if demographic:
                if feature == 'all':
                    data = pd.concat([fmri_df, env_df, new_demo_df], axis=1)
                elif feature == 'fmri':
                    data = pd.concat([fmri_df, new_demo_df], axis=1)
                elif feature == 'env':
                    data = pd.concat([env_df, new_demo_df], axis=1)
            else:
                if feature == 'all':
                    data = pd.concat([fmri_df, env_df], axis=1)
                elif feature == 'fmri':
                    data = pd.concat([fmri_df], axis=1)
                elif feature == 'env':
                    data = pd.concat([env_df], axis=1)
            
            features_name = data.columns
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            data_scaled = pd.DataFrame(data_scaled, columns=features_name)

            tuning_results_snr_modularity = np.zeros((len(perc_overlap), len(n_cubes)))
            tuning_results_avg_modularity = np.zeros((len(perc_overlap), len(n_cubes)))
            tuning_results_std_modularity = np.zeros((len(perc_overlap), len(n_cubes)))

            bootstrap_samples = get_bootstrap_samples(df=data_scaled, n_bootstrap=args.n_bootstrap, ratio=args.bootstrap_ratio, seed_value=seed_value)

            for i, resolution in enumerate(perc_overlap):
                for j, gain in enumerate(n_cubes):
                    print(f'===== Running with overlap: {resolution}, n_cubes: {gain} =====')

                    modularity_list = []
                    for k in range(len(bootstrap_samples)):
                        # Create TDA graph
                        tdagraph = TDAGraph(
                            X=bootstrap_samples[k],
                            filter_func=filter_,
                            n_cubes=gain,
                            perc_overlap=resolution,
                            cluster_algorithm=args.clusterer,
                            min_samples=args.min_samples,
                            metric=args.metric,
                            eps=args.eps,
                        )
                        graph = tdagraph.create_graph()

                        community = Community(graph[0],
                                              algorithm='louvain',
                                              seed_value=seed_value,
                                              dataset=pd.DataFrame(bootstrap_samples[k]))
                        comms = community.community_detection(k=10)
                        modularity = community.compute_modularity()
                        modularity_list.append(modularity)
                
                    avg_modularity = np.array(modularity_list).mean()
                    std_modularity = np.array(modularity_list).std()
                    print(f'Avg modularity: {avg_modularity}')
                    print(f'Std modularity: {std_modularity}')
                    
                    if avg_modularity <= 0.3:
                        snr_modularity = 0
                    else:
                        snr_modularity = avg_modularity / std_modularity
                    print(f'SNR modularity: {snr_modularity}')

                    tuning_results_snr_modularity[i, j] = snr_modularity
                    tuning_results_avg_modularity[i, j] = avg_modularity
                    tuning_results_std_modularity[i, j] = std_modularity

            # Save results
            perc_overlap_values = ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
            n_cubes_values = ['10', '20', '30', '40', '50', '60']

            heatmap_composite_score = Heatmap(
                matrix=tuning_results_snr_modularity,
                cmap='viridis',
                perc_overlap_values=perc_overlap_values,
                n_cubes_values=n_cubes_values,
            )
            heatmap_composite_score.save(output_path=f'{args.output}/snr_composite_score_heatmap_{filter_}_{args.metric}_{feature}.png')

            heatmap_composite_score = Heatmap(
                matrix=tuning_results_avg_modularity,
                cmap='viridis',
                perc_overlap_values=perc_overlap_values,
                n_cubes_values=n_cubes_values,
            )
            heatmap_composite_score.save(output_path=f'{args.output}/avg_modularity_score_heatmap_{filter_}_{args.metric}_{feature}.png')

            heatmap_composite_score = Heatmap(
                matrix=tuning_results_std_modularity,
                cmap='viridis',
                perc_overlap_values=perc_overlap_values,
                n_cubes_values=n_cubes_values,
            )
            heatmap_composite_score.save(output_path=f'{args.output}/std_modularity_score_heatmap_{filter_}_{args.metric}_{feature}.png')

if __name__ == '__main__':
    parser = ArgumentParser(description='Cover parameters tuning')
    parser.add_argument('--fmri', type=str, required=True, help='Path to fMRI data')
    parser.add_argument('--env', type=str, required=True, help='Path to environment data')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels data')
    parser.add_argument('--demo_vars', type=str, default=None, help='Path to demographic variables data (optional)')

    parser.add_argument('--metric', type=str, default='chebyshev', help='Distance metric to use')
    parser.add_argument('--clusterer', type=str, default='dbscan', help='Clusterer to use')
    parser.add_argument('--min_samples', type=int, default=3, help='Minimum samples for clustering')
    parser.add_argument('--eps', type=float, default=8, help='Epsilon for clustering')

    parser.add_argument('--n_bootstrap', type=int, default=10, help='Number of bootstrap iterations')
    parser.add_argument('--bootstrap_ratio', type=float, default=0.7, help='Ratio of data to use in bootstrap samples')

    parser.add_argument('--output', type=str, required=True, help='Path to output directory')

    args = parser.parse_args()

    main(args)