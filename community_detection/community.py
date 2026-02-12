import networkx as nx
import kmapper as km
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

class Community():
    def __init__(self, 
                 graph,
                 algorithm: str,
                 seed_value: int,
                 dataset: pd.DataFrame,
                 labels_df: pd.DataFrame = None):
        super().__init__()
        self.graph = graph
        self.graph_nx = km.adapter.to_nx(graph)
        self.algorithm = algorithm

        self.dataset = dataset
        self.labels_df = labels_df

        self.seed_value = seed_value

    def community_detection(self, k):
        if self.algorithm == 'louvain':
            self.comms = nx.community.louvain_communities(self.graph_nx, seed=self.seed_value)
        elif self.algorithm == 'greedy':
            self.comms = nx.community.greedy_modularity_communities(self.graph_nx)
        elif self.algorithm == 'label_propagation':
            comms = [set(c) for c in nx.community.label_propagation_communities(self.graph_nx)]
            self.comms = comms
        elif self.algorithm == 'fluid':
            comms = [set(c) for c in nx.community.asyn_fluidc(self.graph_nx, k)]
            self.comms = comms
        
        return self.comms
    
    def get_node_idx_communities(self, merging: bool = False):
        '''
        Get the graph node indices for each community.
        If selected_comms_idx is set, merge those communities into one.

        Returns:
        - communities: list of lists, where each inner list contains the node indices of a community.
        '''
        if merging:
            self.communities = []
            # Merge communities with indices in selected_comms_idx
            if hasattr(self, 'selected_comms_idx') and self.selected_comms_idx:
                merged = []
                for idx in self.selected_comms_idx:
                    merged.extend(self.communities[idx])
                # Remove merged communities from the list
                self.communities = [comm for i, comm in enumerate(self.communities) if i not in self.selected_comms_idx]
                # Add the merged community as a new entry
                self.communities.append(merged)
        else:
            self.communities = []
            for comm in self.comms:
                community = [int(idx.split('cube')[1].split('_')[0]) for idx in comm]
                self.communities.append(community)

        return self.communities
    
    def mapped_into_community(self, hypercubes_idx):
        '''
        Map the hypercubes to their respective communities.

        Parameters:
        - hypercubes_idx: list of hypercube indices to be mapped.
        
        Returns:
        - communities_mapped: list of community indices corresponding to the hypercubes.
        '''
        communities_mapped = []
        for i, comm in enumerate(self.communities):
            for idx in hypercubes_idx:
                if idx in comm:
                    communities_mapped.append(i)
        
        return communities_mapped
    
    def compute_modularity(self):
        if self.graph_nx.number_of_edges() == 0:
            mod = np.nan
        else:
            mod = nx.community.modularity(self.graph_nx, self.comms)
        print(f'Modularity: {mod}')
        return mod

    def get_sample_idx(self):
        cluster_to_membership = {name: attributes['membership'] for name, attributes in list(self.graph_nx.nodes(data=True))}

        result_list = []
        for set_of_clusters in self.comms:
            memberships = [cluster_to_membership.get(cluster, []) for cluster in set_of_clusters]
            result_list.append({'cube_clusters': set_of_clusters, 'memberships': memberships})

        distinct_memberships = []
        for i in range(len(result_list)):
            membership = result_list[i]['memberships']
            distinct_memberships.append(membership)

        final_list = []
        for element in distinct_memberships:
            unique_tuples = {tuple(inner_list) for inner_list in element}
            unique_lists = [list(unique_tuple) for unique_tuple in unique_tuples]
            final_list.append(unique_lists)

        flat_list = []
        for i in range(len(final_list)):
            element = [item for sublist in final_list[i] for item in sublist]
            flat_list.append(element)

        self.selected_samples = []
        for j in range(len(flat_list)):
            elements = [self.dataset.iloc[i] for i in flat_list[j]]
            elem = pd.DataFrame(elements)
            self.selected_samples.append(elem)

        for k in range(len(self.selected_samples)):
            self.selected_samples[k] = self.selected_samples[k].drop_duplicates()

        return self.selected_samples
    
    def plot_matrix(self, plot, matrix, matrix_df, cmap, save_path):
        n = len(self.selected_samples)
        if plot:
            plt.figure(figsize=(6, 6))
            plt.imshow(matrix, cmap=cmap, aspect='equal')

            plt.grid(False)

            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f'{val:.3f}', ha='center', va='center', color='white')

            plt.xticks(ticks=np.arange(n), labels=matrix_df.columns, rotation=90)
            plt.yticks(ticks=np.arange(n), labels=matrix_df.index)
            plt.xlabel('Communities (target)')
            plt.ylabel('Communities (reference)')
            plt.colorbar()
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f'Heatmap saved to {save_path}')
            else:
                plt.show()
    
    def community_overlap(self, plot: bool = False, cmap: str = 'viridis', save_path: str = None):
        n = len(self.selected_samples)
        self.overlap_matrix = np.zeros((n, n), dtype=float)

        subject_sets = [set(df.index) for df in self.selected_samples]

        for i in range(n):
            for j in range(n):
                if len(subject_sets[i]) > 0:
                    intersection_size = len(subject_sets[i].intersection(subject_sets[j]))
                    self.overlap_matrix[i, j] = intersection_size / len(subject_sets[i])
                else:
                    self.overlap_matrix[i, j] = np.nan

        overlap_df = pd.DataFrame(
            self.overlap_matrix,
            index=[f'com_{i+1}' for i in range(n)],
            columns=[f'com_{j+1}' for j in range(n)]
        )

        self.plot_matrix(plot, self.overlap_matrix, overlap_df, cmap, save_path)

        # Symmetrize the matrix
        self.overlap_matrix_sym = (self.overlap_matrix + self.overlap_matrix.T) / 2
        np.fill_diagonal(self.overlap_matrix_sym, 1.0)
        overlap_sym_df = pd.DataFrame(
            self.overlap_matrix_sym,
            index=[f'com_{i+1}' for i in range(n)],
            columns=[f'com_{j+1}' for j in range(n)]
        )
        self.plot_matrix(plot, self.overlap_matrix_sym, overlap_sym_df, cmap, save_path)

        return overlap_df, overlap_sym_df
    
    def community_merging(self, threshold: float):
        n = self.overlap_matrix_sym.shape[0]

        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                overlap = self.overlap_matrix_sym[i, j]
                if overlap >= threshold:
                    pairs.append((f'com_{i+1}-com{j+1}', self.overlap_matrix_sym[i, j], i, j))

        df = pd.DataFrame(pairs, columns=['Community Pair', 'Overlap', 'i', 'j'])
        df = df.sort_values(by='Overlap', ascending=False).reset_index(drop=True)

        selected_pairs = []
        used_nodes = set()
        self.merged_dfs = []

        for _, row in df.iterrows():
            if row['i'] not in used_nodes and row['j'] not in used_nodes:
                selected_pairs.append((row['Community Pair'], row['Overlap']))

                merged_df = pd.concat([self.selected_samples[row['i']], self.selected_samples[row['j']]])
                merged_df = merged_df.drop_duplicates()
                self.merged_dfs.append(merged_df)

                used_nodes.update([row['i'], row['j']])

        self.selected_comms_idx = []
        for k in range(n):
            if k not in used_nodes:
                self.merged_dfs.append(self.selected_samples[k])
                selected_pairs.append((f'com_{k+1}', np.nan))
            else:
                self.selected_comms_idx.append(k) 

        result_df = pd.DataFrame(selected_pairs, columns=['Community Pair', 'Overlap'])
        return result_df, self.merged_dfs
    
    def communities_filtering(self, comm_dfs_list):
        subject_to_dfs = defaultdict(set)

        for i, df in enumerate(comm_dfs_list):
            for subject in df.index.unique():
                subject_to_dfs[subject].add(i)

        # Escludi i soggetti CHR
        chr_subjects = set(self.labels_df[self.labels_df.iloc[:, 0] == 'CHR'].index)

        # Individua i soggetti condivisi (escludendo CHR)
        shared_subjects = {
            subject for subject, idxs in subject_to_dfs.items()
            if len(idxs) > 1 and subject not in chr_subjects
        }
        shared_subjects_w_chr = {
            subject for subject, idxs in subject_to_dfs.items()
            if len(idxs) > 1
        }

        print(f'Found {len(shared_subjects)} subjects shared across communities (excluding CHR).')
        print(f'Found {len(shared_subjects_w_chr)} subjects shared across communities (including CHR).')

        # Filtra i soggetti condivisi da ogni comunità
        self.filtered_dfs = [df[~df.index.isin(shared_subjects_w_chr)].copy() for df in comm_dfs_list]

        # Sanity check: soggetti rimasti in tutte le comunità
        all_subjects = [set(df.index) for df in self.filtered_dfs]
        if all_subjects:
            intersection = set.intersection(*all_subjects)
            print('Remaining subjects in all communities:', intersection)
        else:
            print('No communities left after filtering.')

        return self.filtered_dfs
    
    def get_labels_frequency(self, comm_dfs_list):
        labels_columns_names = pd.get_dummies(self.labels_df.iloc[:, 0]).columns.to_list()
        labels_frequency_df = pd.DataFrame(columns=labels_columns_names)

        for i in range(len(comm_dfs_list)):
            df = comm_dfs_list[i]
            labels = self.labels_df.loc[df.index]
            labels_dummy = pd.get_dummies(labels.iloc[:, 0])
            labels_frequency = labels_dummy.sum(axis=0).reindex(labels_columns_names, fill_value=0).to_frame().T
            labels_frequency_df = pd.concat([labels_frequency_df, labels_frequency], axis=0)
        labels_frequency_df.index = [f'com_{i+1}' for i in range(len(labels_frequency_df))]
        labels_frequency_df = labels_frequency_df.fillna(0).astype(int)
        
        return labels_frequency_df
    
    

