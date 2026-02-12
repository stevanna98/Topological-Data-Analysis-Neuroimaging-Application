import numpy as np
import matplotlib.pyplot as plt

class Heatmap:
    def __init__(self, 
                 matrix: np.ndarray,
                 cmap: str = 'viridis',
                 perc_overlap_values: list = None,
                 n_cubes_values: list = None):
        if matrix.ndim != 2:
            raise ValueError('Input matrix must be 2D')
        self.matrix = matrix
        self.cmap = cmap
        self.perc_overlap_values = perc_overlap_values
        self.n_cubes_values = n_cubes_values

    def save(self, 
             output_path: str,
             show_colorbar: bool = True):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.matrix, cmap=self.cmap, aspect='equal')

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                val = self.matrix[i, j]
                if abs(val) < 1e-3 or abs(val) > 1e3:
                    text = f"{val:.2e}"  
                else:
                    text = f"{val:.4f}"  
                plt.text(j, i, text, ha='center', va='center', color='white')

        x_labels = self.n_cubes_values if self.n_cubes_values else [f'res_{i}' for i in range(self.matrix.shape[1])]
        y_labels = self.perc_overlap_values if self.perc_overlap_values else [f'gain_{i}' for i in range(self.matrix.shape[0])]
        plt.xticks(ticks=np.arange(self.matrix.shape[1]), labels=x_labels, rotation=90)
        plt.yticks(ticks=np.arange(self.matrix.shape[0]), labels=y_labels)
        plt.xlabel('Resolution')
        plt.ylabel('Gain')

        if show_colorbar:
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Heatmap saved to {output_path}')

    def show(self,
             show_colorbar: bool = True):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.matrix, cmap=self.cmap, aspect='equal')

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                val = self.matrix[i, j]
                if abs(val) < 1e-3 or abs(val) > 1e3:
                    text = f"{val:.2e}"  
                else:
                    text = f"{val:.4f}"  
                plt.text(j, i, text, ha='center', va='center', color='white')

        x_labels = self.n_cubes_values if self.n_cubes_values else [f'res_{i}' for i in range(self.matrix.shape[1])]
        y_labels = self.perc_overlap_values if self.perc_overlap_values else [f'gain_{i}' for i in range(self.matrix.shape[0])]
        plt.xticks(ticks=np.arange(self.matrix.shape[1]), labels=x_labels, rotation=90)
        plt.yticks(ticks=np.arange(self.matrix.shape[0]), labels=y_labels)
        plt.xlabel('Resolution')
        plt.ylabel('Gain')

        if show_colorbar:
            plt.colorbar()
        plt.tight_layout()