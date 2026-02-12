import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tda.filters import Filter
from tda.tda_graph import TDAGraph

class MappingTDA(TDAGraph):
    def __init__(self,
                 lens, 
                 cover,
                 bins):
        self.data = lens
        self.cover = cover
        
        self.bins = [(sublist[0], sublist[1]) for sublist in bins]
    
    def mapping(self, sample):
        hypercubes = self.cover.transform(self.data, self.bins)

        index_cubes = self.cover.find(sample)

        # Get indices of matching hypercubes
        hypercubes_index = []
        for index in index_cubes:
            cube = self.cover.transform_single(self.data, self.bins[index])

            for j in range(len(hypercubes)):
                if np.array_equal(cube, hypercubes[j]):
                    hypercubes_index.append(j)

        # Return indices, hypercubes, and centers
        return hypercubes_index, hypercubes, self.bins

    def build_contingency_table(self,
                                data,
                                labels,
                                contigency_table_constructed,
                                tdagraph,
                                communities,
                                return_overlap=False,
                                plot_overlap=False,
                                save_path=None,
                                cmap='viridis'):
        
        # Include all groups from labels
        unique_groups = labels["Study Group"].unique()
        group_columns = sorted(unique_groups)
        
        contingency_table_single = pd.DataFrame(
            0,
            index=contigency_table_constructed.index,
            columns=group_columns
        )
        
        subject_to_comms = {}
        no_community_count = 0
        multiple_community_count = 0

        for idx in range(len(data)):
            sample_label = labels.iloc[idx]["Study Group"]
            sample = data.iloc[[idx]]

            # Apply the mapping pipeline
            sample_filtered = tdagraph.sample_filtering(sample)
            hypercubes_index, _, _ = self.mapping(sample_filtered)
            comms_mapped = communities.mapped_into_community(hypercubes_index)

            unique_comms = set(comms_mapped)
            subject_to_comms[idx] = unique_comms

            num_comms = len(unique_comms)
            if num_comms == 0:
                no_community_count += 1
            elif num_comms > 1:
                multiple_community_count += 1
            elif num_comms == 1:
                com = f'com_{list(unique_comms)[0] + 1}'
                contingency_table_single.loc[com, sample_label] += 1

        print(f"Number of samples mapped to no community: {no_community_count}")
        print(f"Number of samples mapped to more than one community: {multiple_community_count}")
        
        overlap_df = None
        if return_overlap:
            n = len(contigency_table_constructed.index)
            overlap_matrix = np.zeros((n, n), dtype=float)

            community_subjects = [set() for _ in range(n)]
            for subj, comms in subject_to_comms.items():
                for comm in comms:
                    community_subjects[comm].add(subj)

            for i in range(n):
                for j in range(n):
                    if len(community_subjects[i]) > 0:
                        intersection_size = len(community_subjects[i].intersection(community_subjects[j]))
                        overlap_matrix[i, j] = intersection_size / len(community_subjects[i])
                    else:
                        overlap_matrix[i, j] = np.nan

            overlap_df = pd.DataFrame(
                overlap_matrix,
                index=[f'com_{i+1}' for i in range(n)],
                columns=[f'com_{j+1}' for j in range(n)]
            )

            if plot_overlap:
                plt.figure(figsize=(6, 6))
                plt.imshow(overlap_matrix, cmap=cmap, aspect="equal")

                plt.grid(False)

                for i in range(n):
                    for j in range(n):
                        val = overlap_matrix[i, j]
                        if not np.isnan(val):
                            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white")

                plt.xticks(ticks=np.arange(n), labels=overlap_df.columns, rotation=90)
                plt.yticks(ticks=np.arange(n), labels=overlap_df.index)
                plt.xlabel("Communities (target)")
                plt.ylabel("Communities (reference)")
                plt.colorbar()
                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=500, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    print(f"Heatmap saved to {save_path}")
                else:
                    plt.show()

        return contingency_table_single, overlap_df