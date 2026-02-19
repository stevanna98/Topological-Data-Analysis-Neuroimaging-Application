import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scikit_posthocs as sp

from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.graphics.mosaicplot import mosaic
from itertools import combinations
from scipy.spatial.distance import jensenshannon

def compute_chi_square_effect_sizes(chi2_stat, contingency_table):
    if hasattr(contingency_table, 'values'):
        observed = contingency_table.values
    else:
        observed = np.array(contingency_table)
    
    # Total sample size
    n = np.sum(observed)
    
    # Number of rows and columns
    n_rows, n_cols = observed.shape
    
    # Degrees of freedom
    df_star = min(n_rows - 1, n_cols - 1)
    
    # 1. Cramér's V
    cramers_v = np.sqrt(chi2_stat / (n * df_star))
    
    # 2. Phi coefficient (same as Cramér's V for 2x2 tables)
    phi = np.sqrt(chi2_stat / n)
    
    # 3. Contingency Coefficient (C)
    contingency_coef = np.sqrt(chi2_stat / (chi2_stat + n))
    
    # 4. Tschuprow's T (alternative to Cramér's V)
    tschuprow_t = np.sqrt(chi2_stat / (n * np.sqrt((n_rows - 1) * (n_cols - 1))))
    
    # Effect size interpretation for Cramér's V
    if df_star == 1:
        if cramers_v < 0.1:
            v_interpretation = "negligible"
        elif cramers_v < 0.3:
            v_interpretation = "small"
        elif cramers_v < 0.5:
            v_interpretation = "medium"
        else:
            v_interpretation = "large"
    elif df_star == 2:
        if cramers_v < 0.07:
            v_interpretation = "negligible"
        elif cramers_v < 0.21:
            v_interpretation = "small"
        elif cramers_v < 0.35:
            v_interpretation = "medium"
        else:
            v_interpretation = "large"
    else:  # df_star >= 3
        if cramers_v < 0.06:
            v_interpretation = "negligible"
        elif cramers_v < 0.17:
            v_interpretation = "small"
        elif cramers_v < 0.29:
            v_interpretation = "medium"
        else:
            v_interpretation = "large"
    
    return {
        'cramers_v': cramers_v,
        'phi': phi,
        'contingency_coefficient': contingency_coef,
        'tschuprow_t': tschuprow_t,
        'cramers_v_interpretation': v_interpretation,
        'sample_size': n,
        'df_star': df_star
    }

def compute_kruskal_wallis_effect_size(h_stat, total_obs):
    if np.isnan(h_stat) or total_obs <= 1:
        return {'epsilon_squared': np.nan, 'interpretation': 'not applicable'}
        
    # Prevent division by zero if total_obs is 1
    if (total_obs - 1) == 0:
        return {'epsilon_squared': np.nan, 'interpretation': 'not applicable (N=1)'}

    epsilon_squared = h_stat / (total_obs - 1)

    # Interpretation (based on Cohen's d benchmarks, adapted)
    if epsilon_squared < 0.01:
        interpretation = "negligible"
    elif epsilon_squared < 0.08:
        interpretation = "small"
    elif epsilon_squared < 0.26:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        'epsilon_squared': epsilon_squared, 
        'interpretation': interpretation
    }

# --- Chi-Squared Analysis Class ---
class ChiSquaredAnalyzer:
    def __init__(self, study_group_frequency: pd.DataFrame, correction_method: str = 'fdr_bh'):
        self.study_group_frequency = study_group_frequency
        self.correction_method = correction_method
        self.study_group_freq_int = self.study_group_frequency.astype(int)

        self.chi2 = None
        self.p_value = None
        self.dof = None
        self.expected = None
        self.residuals_df = None
        self.pvals_corrected_df = None
        self.significant_df = None
        self.effect_size = None

    def run_analysis(self, verbose=False):
        # Step 1: Perform chi-squared test
        try:
            self.chi2, self.p_value, self.dof, self.expected = stats.chi2_contingency(self.study_group_freq_int.values)
            print(f"Global Chi2 = {self.chi2:.2f}, p = {self.p_value}, dof = {self.dof}")

            self.effect_sizes = compute_chi_square_effect_sizes(
                self.chi2, 
                self.study_group_freq_int
            )
            
            print(f"\nEffect Sizes:")
            print(f"  Cramér's V = {self.effect_sizes['cramers_v']:.4f} ({self.effect_sizes['cramers_v_interpretation']})")
            print(f"  Phi = {self.effect_sizes['phi']:.4f}")
            print(f"  Contingency Coefficient = {self.effect_sizes['contingency_coefficient']:.4f}")
            print(f"  Tschuprow's T = {self.effect_sizes['tschuprow_t']:.4f}")
        except ValueError as e:
            print(f"Chi-square failed: {e}")
            self.chi2, self.p_value, self.dof, self.expected = [np.nan]*4
            self.effect_sizes = None

        # Extract basic counts
        observed = self.study_group_freq_int.values
        expected = self.expected
        row_totals = observed.sum(axis=1).reshape(-1, 1)
        col_totals = observed.sum(axis=0).reshape(1, -1)
        N = observed.sum()

        # Step 2: Adjusted standardized residuals
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.sqrt(expected * (1 - row_totals / N) * (1 - col_totals / N))
            adjusted_residuals = (observed - expected) / denom
            adjusted_residuals = np.nan_to_num(adjusted_residuals)

        self.residuals_df = pd.DataFrame(adjusted_residuals,
                                        index=self.study_group_frequency.index,
                                        columns=self.study_group_frequency.columns)

        # Step 3: Per-cell p-values from adjusted residuals
        p_values = 2 * (1 - stats.norm.cdf(np.abs(adjusted_residuals)))
        pvals_df = pd.DataFrame(p_values,
                                index=self.study_group_frequency.index,
                                columns=self.study_group_frequency.columns)

        # Step 4: Correction
        rej, pvals_corrected_flat, _, _ = multipletests(p_values.flatten(), method=self.correction_method)
        pvals_corrected = pvals_corrected_flat.reshape(p_values.shape)
        self.pvals_corrected_df = pd.DataFrame(pvals_corrected,
                                            index=self.study_group_frequency.index,
                                            columns=self.study_group_frequency.columns)

        # Step 5: Significant cells (True/False after correction)
        self.significant_df = pd.DataFrame(rej.reshape(p_values.shape),
                                        index=self.study_group_frequency.index,
                                        columns=self.study_group_frequency.columns)

        if verbose:
            # Output results
            print("\nAdjusted Standardized Residuals:")
            print(self.residuals_df.round(2))
            print("\nPer-cell p-values (before correction):")
            print(pvals_df.round(4))
            print(f'\nCorrection method: {self.correction_method}')
            print(self.pvals_corrected_df.round(4))
            print("\nSignificant Cells (after correction):")
            print(self.significant_df)

    def _get_color_for_mosaic(self, key_path):
        if self.residuals_df is None:
            raise ValueError("Analysis has not been run. Call .run_analysis() first.")

        group, category = key_path
        residual_value = self.residuals_df.loc[group, category]

        if residual_value > 2:
            color = '#4CAF50' 
        elif residual_value > 1:
            color = '#8BC34A' 
        elif residual_value < -2:
            color = '#F44336' 
        elif residual_value < -1:
            color = '#FF9800' 
        else:
            color = '#BBDEFB' 

        return {'color': color}

    def plot_mosaic(self):
        if self.residuals_df is None:
            raise ValueError("Analysis has not been run. Call .run_analysis() first.")

        # Prepare data for the mosaic plot
        df_for_mosaic = self.study_group_freq_int.reset_index().melt(
            id_vars='index', var_name='Category', value_name='Count'
        )
        df_for_mosaic.rename(columns={'index': 'Group'}, inplace=True)

        # Create the figure and axes for the plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Generate the mosaic plot on the created axes
        mosaic(df_for_mosaic,
            index=['Group', 'Category'],
            properties=self._get_color_for_mosaic,
            title='Mosaic Plot of Study Group Frequencies by Category (Residuals Colored)',
            label_rotation=45,
            gap=0.01,
            ax=ax)
        
        # Manually create legend handles (patches) and labels
        legend_patches = [
            mpatches.Patch(color='#4CAF50', label='Significantly Higher (Res > 2)'),
            mpatches.Patch(color='#8BC34A', label='Higher (1 < Res ≤ 2)'),
            mpatches.Patch(color='#F44336', label='Significantly Lower (Res < -2)'),
            mpatches.Patch(color='#FF9800', label='Lower (-2 ≤ Res < -1)'),
            mpatches.Patch(color='#BBDEFB', label='As Expected (-1 ≤ Res ≤ 1)')
        ]
        
        # Add the legend to the plot
        ax.legend(handles=legend_patches, title='Residuals', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Ensure a tight layout to accommodate the legend
        plt.tight_layout()
        plt.show()

    def plot_observed_vs_expected(self):
        if self.expected is None:
            raise ValueError("Analysis has not been run. Call .run_analysis() first.")

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('viridis')

        # 1. Prepare data for plotting
        expected_df = pd.DataFrame(self.expected,
                                   index=self.study_group_frequency.index,
                                   columns=self.study_group_frequency.columns)

        observed_melted = self.study_group_freq_int.reset_index().melt(
            id_vars='index', var_name='Category', value_name='Count'
        )
        observed_melted.rename(columns={'index': 'Group'}, inplace=True)
        observed_melted['Type'] = 'Observed'

        expected_melted = expected_df.reset_index().melt(
            id_vars='index', var_name='Category', value_name='Count'
        )
        expected_melted.rename(columns={'index': 'Group'}, inplace=True)
        expected_melted['Type'] = 'Expected'

        # 2. Create the plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        
        # Plot Observed Frequencies
        sns.barplot(data=observed_melted, x='Group', y='Count', hue='Category',
            ax=axes[0], palette='tab10', dodge='auto')
        axes[0].set_title('Observed Frequencies', fontsize=16)
        axes[0].set_xlabel('Group', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        plt.setp(axes[0].get_xticklabels(), ha='right')
        axes[0].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot Expected Frequencies
        sns.barplot(data=expected_melted, x='Group', y='Count', hue='Category',
                    ax=axes[1], palette='tab10', dodge='auto')
        axes[1].set_title('Expected Frequencies (under independence)', fontsize=16)
        axes[1].set_xlabel('Group', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        plt.setp(axes[1].get_xticklabels(), ha='right')
        axes[1].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    def community_entropy(self):
        community_entropies = []
        majority_classes = []
        for idx, row in self.study_group_frequency.iterrows():
            label_counts = row.values
            label_probs = label_counts / label_counts.sum()
            ent = stats.entropy(label_probs, base=2)
            community_entropies.append(ent)
            majority_classes.append(row.idxmax())

        community_entropy_df = pd.DataFrame({
            'Community': self.study_group_frequency.index,
            'Entropy': community_entropies,
            'Majority_Class': majority_classes
        }).set_index('Community')

        return community_entropy_df

# --- Feature Set Statistics Analysis Class ---
class FeatureSetStatisticsAnalyzer:
    def __init__(self, labels, study_group_frequency, correction_method='fdr_bh', alpha=0.05):
        self.labels = labels
        self.community_names = study_group_frequency.index.tolist()
        self.correction_method = correction_method
        self.alpha = alpha

        # Infer group names from labels
        self.label_names = self.labels['Study Group'].unique()
        self.group_combinations = list(combinations(self.label_names, 2))

        # Initialize result attributes
        self.global_stats_df = None
        self.significant_global_features = None
        self.community_stats_results = {}

    def compute_feature_correlations(self, x, x_embedded, feature_names=None, method='pearson', ratio=0.3):
        if isinstance(x, pd.DataFrame):
            x = np.array(x)
        
        n_features = x.shape[1]
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        corr_func = stats.pearsonr if method == 'pearson' else stats.spearmanr
        results = []

        for i in range(n_features):
            feature_data = x[:, i]
            corr_1, pval_1 = corr_func(feature_data, x_embedded[:, 0])
            corr_2, pval_2 = corr_func(feature_data, x_embedded[:, 1])

            # Pick the component with the strongest absolute correlation
            best_component = 1 if abs(corr_1) >= abs(corr_2) else 2
            best_corr = corr_1 if best_component == 1 else corr_2
            best_pval = pval_1 if best_component == 1 else pval_2

            results.append({
                'feature': feature_names[i],
                'feature_idx': i,
                'corr_filter_1': corr_1,
                'pval_filter_1': pval_1,
                'corr_filter_2': corr_2,
                'pval_filter_2': pval_2,
                'best_component': best_component,
                'best_corr': best_corr,
                'best_pval': best_pval
            })

        results_df = pd.DataFrame(results)

        # Multiple comparison correction if specified
        if self.correction_method is not None:
            reject, pvals_corrected, _, _ = multipletests(
                results_df['best_pval'].values,
                method=self.correction_method
            )
            results_df['pval_corrected'] = pvals_corrected
            results_df['significant'] = reject
            results_df['final_pval'] = results_df['pval_corrected']
        else:
            results_df['significant'] = results_df['best_pval'] < self.alpha
            results_df['final_pval'] = results_df['best_pval']

        # Select features significant with at least one component
        selected_df = results_df[results_df['significant']].copy()

        # --- NEW STEP: If fewer than 100 features, keep top 30% (by p-value)
        if len(selected_df) > 100:
            n_keep = 10
            selected_df = selected_df.nsmallest(n_keep, 'final_pval')
        else:
            n_keep = max(1, int(len(selected_df) * ratio))
            selected_df = selected_df.nsmallest(n_keep, 'final_pval')

        # Sort for clarity
        selected_df = selected_df.sort_values(by='final_pval')
        selected_features = selected_df['feature'].tolist()

        return results_df, selected_features
    
    def plot_correlation_results(self, results_df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Correlation with UMAP dimension 1
        ax = axes[0, 0]
        colors = ['red' if p < self.alpha else 'gray' for p in results_df['pval_umap1']]
        ax.scatter(range(len(results_df)), results_df['corr_umap1'], 
                c=colors, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Correlation with UMAP Dim 1')
        ax.set_title('Correlations with UMAP Dimension 1')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Correlation with UMAP dimension 2
        ax = axes[0, 1]
        colors = ['red' if p < self.alpha else 'gray' for p in results_df['pval_umap2']]
        ax.scatter(range(len(results_df)), results_df['corr_umap2'], 
                c=colors, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Correlation with UMAP Dim 2')
        ax.set_title('Correlations with UMAP Dimension 2')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: P-values distribution
        ax = axes[1, 0]
        ax.hist(results_df['min_pval'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=self.alpha, color='red', linestyle='--', linewidth=2, 
                label=f'α = {self.alpha}')
        ax.set_xlabel('Minimum P-value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Minimum P-values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Max absolute correlation vs min p-value
        ax = axes[1, 1]
        significant = results_df['significant']
        ax.scatter(results_df[~significant]['max_abs_corr'], 
                results_df[~significant]['min_pval'],
                c='gray', alpha=0.6, label='Not significant')
        ax.scatter(results_df[significant]['max_abs_corr'], 
                results_df[significant]['min_pval'],
                c='red', alpha=0.6, label='Significant')
        ax.axhline(y=self.alpha, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Maximum Absolute Correlation')
        ax.set_ylabel('Minimum P-value')
        ax.set_title('Correlation Strength vs Significance')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def run_global_analysis(self, samples_df):
        print("--- Running Global Statistical Analysis (Kruskal–Wallis / Chi-square) ---")
        stats_results = []
        
        if not samples_df:
            print("Warning: No sample dataframes provided. Skipping global analysis.")
            return

        feature_names = samples_df[0].columns

        for feature in feature_names:
            community_values = [df[feature].values for df in samples_df]

            # Filter out empty or constant groups
            valid_community_values = [g for g in community_values if len(g) > 1 and len(np.unique(g)) > 1]

            effect_size_val = np.nan
            effect_size_interp = 'N/A'

            if len(valid_community_values) < 2:
                stats_results.append({
                    'Feature': feature,
                    'Test_Used': 'None',
                    'Statistic': np.nan,
                    'p_value': np.nan,
                    'effect_size_val': effect_size_val,
                    'effect_size_interp': effect_size_interp
                })
                continue

            # --- Chi-square for categorical features ---
            if feature in ['immigration_yes', 'SEX']:
                # Build contingency table
                contingency_table = pd.DataFrame({
                    f'Community_{i}': df[feature].value_counts() for i, df in enumerate(samples_df)
                }).fillna(0)
                
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency_table)
                    test_used = 'Chi-square'
                    effect_sizes = compute_chi_square_effect_sizes(chi2_stat, contingency_table)
                    effect_size_val = effect_sizes['cramers_v']
                    effect_size_interp = effect_sizes['cramers_v_interpretation']
                else:
                    chi2_stat, p_val, test_used = np.nan, np.nan, 'Chi-square (invalid table)'

                stats_results.append({
                    'Feature': feature,
                    'Test_Used': test_used,
                    'Statistic': chi2_stat,
                    'p_value': p_val,
                    'effect_size_val': effect_size_val,
                    'effect_size_interp': effect_size_interp
                })
                continue

            # --- Kruskal–Wallis for all other features ---
            try:
                test_stat, p_val = stats.kruskal(*valid_community_values)
                test_used = 'Kruskal–Wallis'

                total_obs = sum(len(g) for g in valid_community_values)
                kw_results = compute_kruskal_wallis_effect_size(test_stat, total_obs)
                effect_size_val = kw_results['epsilon_squared']
                effect_size_interp = kw_results['interpretation']
            except ValueError:
                test_stat, p_val, test_used = np.nan, np.nan, 'Kruskal–Wallis (error)'

            stats_results.append({
                'Feature': feature,
                'Test_Used': test_used,
                'Statistic': test_stat,
                'p_value': p_val,
                'effect_size': effect_size_val,
                'effect_size_interp': effect_size_interp    
            })

        # Create results DataFrame
        self.global_stats_df = pd.DataFrame(stats_results)

        # Sort and extract significant features
        self.global_stats_df_sorted = self.global_stats_df.sort_values('p_value')
        self.significant_global_features = self.global_stats_df_sorted[
            self.global_stats_df_sorted['p_value'] < self.alpha
        ]

        print(f"Number of significant features: {len(self.significant_global_features)}")
        print("\nMost significant features (Global):")
        print(self.significant_global_features[[
            'Feature', 'Test_Used', 'Statistic', 'p_value', 'effect_size', 'effect_size_interp'
        ]])

    def run_global_analysis_with_filtering(self, samples_df, categorical_features=['immigration_yes','SEX'], variance_threshold=0.1):
        print("--- Running Global Statistical Analysis with Filtering ---")
        if not samples_df or len(samples_df) == 0:
            print("Warning: No sample dataframes provided. Skipping analysis.")
            return

        numerical_features = [f for f in samples_df[0].columns if f not in categorical_features]
        variances = pd.concat([df[numerical_features] for df in samples_df], axis=0).var()
        filtered_numerical_features = variances[variances > variance_threshold].index.tolist()

        all_features = filtered_numerical_features + categorical_features
        print(f'Number of features after variance filtering (thr={variance_threshold}): {len(all_features)}')

        stats_results = []

        for feature in all_features:
            community_values = [df[feature].values for df in samples_df]
            valid_groups = [g for g in community_values if len(g) > 1 and len(np.unique(g)) > 1]

            if len(valid_groups) < 2:
                stats_results.append({'Feature': feature, 'Test_Used': 'None', 'Statistic': np.nan, 'p_value': np.nan})
                continue

            if feature in categorical_features:
                contingency_table = pd.DataFrame({
                    f'Community_{i}': df[feature].value_counts() for i, df in enumerate(samples_df)
                }).fillna(0)
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    stat, p, _, _ = stats.chi2_contingency(contingency_table)
                    test_used = 'Chi-square'
                else:
                    stat, p, test_used = np.nan, np.nan, 'Chi-square (invalid table)'
                stats_results.append({'Feature': feature, 'Test_Used': test_used, 'Statistic': stat, 'p_value': p})
                continue

            try:
                stat, p = stats.kruskal(*valid_groups)
                test_used = 'Kruskal-Wallis'
            except ValueError:
                stat, p, test_used = np.nan, np.nan, 'Kruskal-Wallis (error)'

            stats_results.append({'Feature': feature, 'Test_Used': test_used, 'Statistic': stat, 'p_value': p})

        results_df = pd.DataFrame(stats_results)
        valid_pvals = results_df['p_value'].fillna(1).values
        _, pvals_fdr, _, _ = multipletests(valid_pvals, alpha=self.alpha, method='fdr_bh')
        results_df['p_value_fdr'] = pvals_fdr
        results_df['significant'] = results_df['p_value_fdr'] < self.alpha

        self.global_stats_df = results_df
        self.significant_global_features = results_df[results_df['significant']].sort_values('p_value_fdr')

        print(f"Number of significant features after FDR: {len(self.significant_global_features)}")
        print(self.significant_global_features[['Feature','Test_Used','Statistic','p_value','p_value_fdr']])

class ContingencyTableComparator:
    def __init__(self,
                 table_1: np.ndarray,
                 table_2: np.ndarray,
                 random_state=None):
        # Ensure both tables are numpy arrays
        if not isinstance(table_1, np.ndarray):
            table_1 = np.array(table_1)
        if not isinstance(table_2, np.ndarray):
            table_2 = np.array(table_2)
        self.table_1 = table_1
        self.table_2 = table_2

        self.rng = np.random.default_rng(random_state)

        self.table_1_flat = self.table_1.flatten()
        self.table_2_flat = self.table_2.flatten()

    def compute_stats(self, table_1=None, table_2=None):
        if table_1 is None and table_2 is None:
            # Cosine Similarity
            cosine_sim = np.dot(self.table_1_flat, self.table_2_flat) / (np.linalg.norm(self.table_1_flat) * np.linalg.norm(self.table_2_flat))

            p = self.table_1_flat / np.sum(self.table_1_flat)
            q = self.table_2_flat / np.sum(self.table_2_flat)
        else:
            table_1_flat = table_1.flatten()
            table_2_flat = table_2.flatten()

            cosine_sim = np.dot(table_1_flat, table_2_flat) / (np.linalg.norm(table_1_flat) * np.linalg.norm(table_2_flat))

            p = table_1_flat / np.sum(table_1_flat)
            q = table_2_flat / np.sum(table_2_flat)

        # Jensen-Shannon Divergence
        js_div = jensenshannon(p, q)

        # Bhattacharyya Distance
        bc = np.sum(np.sqrt(p * q))
        bhatt_dist = -np.log(bc)

        return {
            'cosine_similarity': cosine_sim,
            'jensen_shannon_divergence': js_div,
            'bhattacharyya_distance': bhatt_dist
        }

    def permutation_test(self, n_permutations=1000):
        observed = self.compute_stats()

        n_rows = self.table_1.shape[0]

        null_cosine, null_js, null_bhatt = [], [], []

        for _ in range(n_permutations):
            # Stack rows from both tables
            all_rows = np.vstack([self.table_1, self.table_2])

            # Shuffle the rows
            self.rng.shuffle(all_rows)

            # Split back into permuted tables
            perm_table_1 = all_rows[:self.table_1.shape[0], :]
            perm_table_2 = all_rows[self.table_1.shape[0]:, :]

            # Flatten before computing stats (needed for cosine, JS, Bhattacharyya)
            stats = self.compute_stats(
                perm_table_1.flatten(),
                perm_table_2.flatten()
            )

            null_cosine.append(stats['cosine_similarity'])
            null_js.append(stats['jensen_shannon_divergence'])
            null_bhatt.append(stats['bhattacharyya_distance'])

        null_cosine = np.array(null_cosine)
        null_js = np.array(null_js)
        null_bhatt = np.array(null_bhatt)

        # Permutation p-values
        p_cosine = (np.sum(null_cosine >= observed['cosine_similarity']) + 1) / (n_permutations + 1)
        p_js = (np.sum(null_js <= observed['jensen_shannon_divergence']) + 1) / (n_permutations + 1)
        p_bhatt = (np.sum(null_bhatt >= observed['bhattacharyya_distance']) + 1) / (n_permutations + 1)

        return {
            'observed': observed,
            'p_values': {
                'cosine_similarity': p_cosine,
                'jensen_shannon_divergence': p_js,
                'bhattacharyya_distance': p_bhatt
            },
            'null_distributions': {
                'cosine_similarity': null_cosine,
                'jensen_shannon_divergence': null_js,
                'bhattacharyya_distance': null_bhatt
            }
        }