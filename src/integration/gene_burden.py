import pandas as pd
import numpy as np
from kneed import KneeLocator, find_shape
from scipy.stats import binom
from itertools import combinations

class MutationBurdenAnalyzer:
    """
    Analyzes and identifies genes with excess mutation burden in responsive cell lines
    and extracts co-mutation network edgelists across cell lines for considered pathway.
    """
    def __init__(self, d_res_df, total_zscore_tbl, dose_data_tbl, all_wes_mutation_df, gene_set_dict):
        self.d_res_df = d_res_df
        self.total_zscore_tbl = total_zscore_tbl
        self.dose_data_tbl = dose_data_tbl
        self.all_wes_mutation_df = all_wes_mutation_df
        self.gene_set_dict = gene_set_dict
    def _find_shape_placeholder(self, x, y):
        """Internal placeholder for find_shape logic. Replace with your custom function."""
        kneed_tbl = pd.DataFrame({'x':x,'y':y}).sort_values('x')
        direction, curve = find_shape(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy())
       # Assuming typical curve properties for genomics knee-detections
        return direction, curve
    def analyze_pathway(self, pathway_id, drug_id):
        """
        Executes the mutation enrichment pipeline for a specific pathway-drug combination.
        Parameters:
            pathway_id (str): Identifier for the target gene set/pathway.
            drug_id (int/str): Identifier for the therapeutic compound.
        Returns:
            enriched_genes_df (pd.DataFrame): Outlying responsive gene-cell instances.
            edgelist_df (pd.DataFrame): Pairs of co-mutated genes within the same cell lines.
        """
        # 1. Fetch thresholds and parameters
        pathway_meta = self.d_res_df.query('Gene_Set == @pathway_id')
        if pathway_meta.empty:
            raise ValueError(f"Pathway '{pathway_id}' not found in results dataframe.")
            
        z_thresh = pathway_meta.z_thresh.values[0]
        cutoff = int(pathway_meta.cut_off.values[0])
        
        # 2. Segment Cell Lines (Responsive vs Unresponsive)
        path_df = self.total_zscore_tbl.query('Gene_Set == @pathway_id')
        original_responders = path_df.query('z_score > @z_thresh').cell_id.to_list()
        
        drug_dose_df = self.dose_data_tbl.query('DRUG_ID == @drug_id').sort_values('sensitivity_p')
        
        leading_edge_cells = (
            drug_dose_df.iloc[:cutoff, :]
            .assign(in_pathway=lambda df: np.where(df.SANGER_MODEL_ID.isin(original_responders), 1, 0))
            .query('in_pathway > 0')
            .SANGER_MODEL_ID.to_list()
        )
        unresponsive_cells = drug_dose_df.iloc[cutoff:, :].SANGER_MODEL_ID.to_list()
        all_monitored_cells = unresponsive_cells + leading_edge_cells
        # 3. Calculate Background Null Mutation Rates
        total_mut_count_tbl = (
            self.all_wes_mutation_df.query('sanger_model_id in @all_monitored_cells')
            .groupby('sanger_model_id')
            .agg(mtot=('POS', 'count'))
            .reset_index()
        )
        unresponsive_total_muts = self.all_wes_mutation_df.query('sanger_model_id in @unresponsive_cells').POS.count()
        unresponsive_gene_stats = (
            self.all_wes_mutation_df.query('sanger_model_id in @unresponsive_cells')
            .groupby('gene')
            .agg(mcount=('POS', 'count'))
            .reset_index()
            .assign(pnull=lambda df: df.mcount / unresponsive_total_muts)
        )
        # 4. Construct Master Genomic Matrix
        obs_mutations = (
            self.all_wes_mutation_df.query('sanger_model_id in @all_monitored_cells')
            .groupby(['sanger_model_id', 'gene'])
            .agg(obs=('POS', 'count'))
            .reset_index()
        )
        gene_matrix_df = (
            pd.DataFrame({'gene': self.gene_set_dict[pathway_id]})
            .merge(pd.DataFrame({'sanger_model_id': all_monitored_cells}), how='cross')
            .merge(total_mut_count_tbl, on='sanger_model_id', how='left')
            .merge(unresponsive_gene_stats[['gene', 'pnull']], on='gene', how='left')
            .merge(obs_mutations, on=['sanger_model_id', 'gene'], how='left')
            .fillna(0)
        )
        # 5. Statistical Inference (Binomial Survival Function)
        active_genes = gene_matrix_df.query('sanger_model_id in @leading_edge_cells & obs > 0').gene.unique()
        tested_matrix_df = (
            gene_matrix_df.query('gene in @active_genes')
            .assign(
                binomp=lambda df: binom.sf(df.obs - 1, df.mtot, df.pnull),
                kind=lambda df: np.where(df.sanger_model_id.isin(leading_edge_cells), 'responsive', 'unresponsive'),
                m_or=lambda df: (df.obs / df.mtot) / df.pnull
            )
        )
        # 6. Knee Point Outlier Flagging (Iterative per Gene)
        enriched_dfs = []
        for target_gene in tested_matrix_df.gene.unique():
            gene_subset = tested_matrix_df.query('gene == @target_gene & obs > 0').copy()
            if len(gene_subset) < 3:  # Knee detection requires statistical distribution space
                continue
            gene_subset = gene_subset.assign(
                pscore=lambda df: -np.log10(df.binomp),
                pcr=lambda df: df.binomp.rank(pct=True)
            ).sort_values('binomp')
            # Use internal shape detection or clean default
            try:
                direction, curve = self._find_shape_placeholder(gene_subset.pcr.to_numpy(), gene_subset.pscore.to_numpy())
                kl = KneeLocator(gene_subset.pcr.to_numpy(), gene_subset.pscore.to_numpy(), curve=curve, direction=direction)
                if kl.knee is not None:
                    enriched_dfs.append(gene_subset.query('pcr < @kl.knee & kind == "responsive"'))
            except Exception:
                continue # Safeguard pipeline against mathematical convergence edgecases
        if not enriched_dfs:
            return pd.DataFrame(), pd.DataFrame()
        enriched_genes_df = pd.concat(enriched_dfs).assign(Gene_Set=pathway_id)
        # 7. Generate Co-Mutation Edgelist Matrix
        edge_cell_candidates = enriched_genes_df.sanger_model_id.value_counts().reset_index()
        edge_cells = edge_cell_candidates.query('count > 1').sanger_model_id.to_list()
        if not edge_cells:
            return enriched_genes_df, pd.DataFrame()
        gene_modules = (
            enriched_genes_df.query('sanger_model_id in @edge_cells')
            .groupby('sanger_model_id')
            .agg(gene_vertices=('gene', 'unique'))
            .reset_index()
        )
        edge_dfs = []
        for row in gene_modules.itertuples():
            pairs = list(combinations(row.gene_vertices, 2))
            edge_dfs.append(
                pd.DataFrame(pairs, columns=['Gene_1', 'Gene_2'])
                .assign(sanger_model_id=row.sanger_model_id)
            )
        edgelist_df = pd.concat(edge_dfs).assign(
            Gene_Set=pathway_id, 
            leading_edge_n=len(leading_edge_cells)
        )
        return enriched_genes_df, edgelist_df


