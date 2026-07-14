import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import reload
from scipy.stats import beta
from scipy.stats import binom
import xlmhglite
import networkx as nx
import src.utils.io 
import src.mutation.gene_set_analysis
import src.dose_response.detect_response
import src.integration.gene_burden
import src.integration.leading_edge

reload(src.integration.gene_burden)
reload(src.mutation.gene_set_analysis)
reload(src.dose_response.detect_response)
reload(src.utils.io)

# %%
vcf_folder = Path("/home/vipink/Documents/dose_response_workflow/data/omics/mutations_wes_vcf_20250226/")
vcf_file_list = list(vcf_folder.glob("*.gz"))

#gene_set_file = "/home/vipink/Documents/dose_response_workflow/data/metadata/c6.all.v2026.1.Hs.symbols.gmt"

gene_set_file = "/home/vipink/Documents/dose_response_workflow/data/metadata/c2.all.v2026.1.Hs.symbols.gmt"

dose_fit_file = "/home/vipink/Documents/dose_response_workflow/data/GDSC2_fitted_dose_response_27Oct23.csv"
# %%

gene_set_dict = src.utils.io.parse_gmt(gene_set_file)
Gene_Set_size_tbl = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).explode('Genes').Gene_Set.value_counts().reset_index().rename(columns={'count':'gene_count'})

sub_collection_list = ['REACTOME','KEGG','PID','WP']
collection_to_use_list = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).assign(collection = lambda df: [i.split('_')[0] for i in df.Gene_Set]).query('collection in @sub_collection_list').Gene_Set.drop_duplicates().to_list()

gene_set_to_use_dict = {k: gene_set_dict[k] for k in collection_to_use_list if k in gene_set_dict}

# %%
dose_coef_tbl = pd.read_csv(dose_fit_file,sep='\t')
dose_data_tbl = dose_coef_tbl.loc[:,['SANGER_MODEL_ID','CANCER_TYPE', 'DRUG_ID', 'DRUG_NAME','MIN_CONC', 'MAX_CONC','LN_IC50','AUC', 'RMSE']].query('RMSE < 0.2')


null_dose_data_tbl = dose_data_tbl.assign(LN_MAX_CONC = lambda df: np.log(df.MAX_CONC)).assign(inert = lambda df: df.LN_IC50.gt(df.LN_MAX_CONC)).query('inert')

drug_beta_param_df = src.dose_response.detect_response.get_shrunk_beta_params(null_dose_data_tbl)

dose_data_tbl = dose_data_tbl.merge(
        drug_beta_param_df[['DRUG_ID','alpha', 'beta']], 
        how='left')
# 2. Vectorized CDF calculation
# This gives the probability of observing 'auc' or lower given the Null
dose_data_tbl = dose_data_tbl.assign(sensitivity_p = lambda df: beta.cdf(df['AUC'], df['alpha'], df['beta']))
# 3. For ranking, we often use the Negative Log 10 of the probability
# This makes 'stronger' hits have higher positive values
dose_data_tbl = dose_data_tbl.assign(rank_score = lambda df: -np.log10(df.sensitivity_p + 1e-100),adjusted_auc_rank = lambda df: df.sensitivity_p.rank(pct=True)) # Avoid log(0)

# %%

all_wes_mutation_df = pd.concat([src.utils.io.get_vcf_summary_tbl(vcf_file) for vcf_file in vcf_file_list]).drop_duplicates()

# %%
# Calculate fixed total exome mutation burden per cell line (Trials)
print("Step 1: Pivoting mutation counts and building core exome vectors...")
total_exome_loads = all_wes_mutation_df.groupby('sanger_model_id').size().rename('total_cell_muts')
base_matrix = all_wes_mutation_df.groupby(['sanger_model_id', 'gene']).size().unstack(fill_value=0)

all_cells = base_matrix.index.tolist()
all_genes = base_matrix.columns.tolist()

# Core numpy structures
K_matrix = base_matrix.values  # Shape: (n_cells, n_genes)
N_vector = total_exome_loads.reindex(all_cells).fillna(0).values.reshape(-1, 1) # Shape: (n_cells, 1)

# %%
#2 Lapatinib = 1558
#3 Vorinostat = 1012
# Gefitinib = 1010
#1 Trametinib = 1372 
drug_id = 1012

tmp_drug_excess_mutation_count_tbl = src.mutation.gene_set_analysis.get_excess_mutation_count_matrix(drug_id,K_matrix,N_vector,dose_data_tbl,all_cells,all_genes)

# %%

gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_excess_mutation_count_tbl,all_cells,all_genes,gene_set_to_use_dict)

# %%

tmp_res = src.mutation.gene_set_analysis.run_high_throughput_parallel_xlmhg(
    pathway_burden_df = gene_set_collection_excess_count_df,   
    drug_sensitivity_df = dose_data_tbl.query('DRUG_ID == @drug_id'), 
    n_burden_steps = 20,
    auc_col = 'sensitivity_p',
    sanger_id_col = 'SANGER_MODEL_ID',
    n_jobs = 8  # Use all available CPU cores
)

# %%
from kneed import KneeLocator, find_shape
kneed_tbl = tmp_res.assign(x = lambda df: df.Min_mHG_P_Value.rank(pct=True),y=lambda df:-np.log10(df.Min_mHG_P_Value)).sort_values('x').loc[:,['Pathway_Name','x','y']]


direction, curve = find_shape(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy())
kl = KneeLocator(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy(), curve=curve, direction=direction)
tmp_ax = kneed_tbl.plot(x='x',y='y')
tmp_ax.axvline(x=kl.knee, color='red', linestyle='--', linewidth=2, label='Threshold')
plt.show()
# %%
# Extract the pathways that constitute outlier association with the drug response
out_path = kneed_tbl.query('x <= @kl.knee').Pathway_Name.to_list()
tmp_gene_set_name = out_path[33]
tmp_gene_set = gene_set_to_use_dict[tmp_gene_set_name]

tmp_gene_set_leading_edge_list = tmp_res.query('Pathway_Name == @tmp_gene_set_name').Leading_Edge_Cell_Lines[0]

dose_data_tbl.query('DRUG_ID == @drug_id and SANGER_MODEL_ID in @tmp_gene_set_leading_edge_list').AUC

gene_set_collection_excess_count_df.loc[tmp_gene_set_leading_edge_list,tmp_gene_set_name]

tmp_drug_excess_mutation_count_tbl.query('sanger_model_id in @tmp_gene_set_leading_edge_list').assign(in_tmp_set = lambda df: df.gene.isin(tmp_gene_set)).query('in_tmp_set')

tmp_drug_excess_mutation_count_tbl.query('sanger_model_id in @tmp_gene_set_leading_edge_list').assign(in_tmp_set = lambda df: df.gene.isin(tmp_gene_set)).query('in_tmp_set').groupby('gene').agg(ncell = ('sanger_model_id','nunique'),s_em = ('excess_mutation_count','sum')).reset_index().assign(lead_prop = lambda df: df.ncell.div(float(len(tmp_gene_set_leading_edge_list))),avg_m = lambda df: df.s_em.div(float(len(tmp_gene_set_leading_edge_list))),Pathwat_Name = tmp_gene_set_name).sort_values('avg_m')

# %%

import numpy as np
import pandas as pd
import itertools

def construct_leading_edge_network(
    pathway_genes: list,
    leading_edge_cells: list,
    excess_mutation_df: pd.DataFrame # Index: sanger_model_id, Columns: genes
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs node and edge tables for the leading edge gene network.
    
    excess_mutation_df should be the pivot/matrix form of your excess scores 
    filtered/reindexed to contain only (leading_edge_cells x pathway_genes).
    """
    # 1. Align and slice the matrix to only include our active sub-space
    # Fill NaN with 0.0 to ensure safe math
    sub_matrix = excess_mutation_df.reindex(
        index=leading_edge_cells, 
        columns=pathway_genes
    ).fillna(0.0).values 
    
    n_cells, n_genes = sub_matrix.shape
    if n_cells == 0 or n_genes == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Total pathway burden per cell line (Sum across columns)
    # Shape: (n_cells, 1) to allow broadcasting
    cell_pathway_totals = np.sum(sub_matrix, axis=1, keepdims=True)
    # Avoid division by zero for cell lines with zero overall burden
    cell_pathway_totals[cell_pathway_totals == 0.0] = 1.0 
    
    # --- Compute Node Weights ---
    # Prevalence: % of cells where score > 0
    node_prevalence = np.mean(sub_matrix > 0.0, axis=0)
    # Intensity: Average excess score over the leading edge
    node_intensity = np.mean(sub_matrix, axis=0)
    
    node_records = []
    for g_idx, gene in enumerate(pathway_genes):
        node_records.append({
            'Gene': gene,
            'Weight_Prevalence': node_prevalence[g_idx],
            'Weight_Intensity': node_intensity[g_idx]
        })
    nodes_df = pd.DataFrame(node_records)
    
    # --- Compute Edge Weights ---
    edge_accumulator = []
    
    # Generate all unique gene pairs within the pathway
    for i, j in itertools.combinations(range(n_genes), 2):
        gene_i = pathway_genes[i]
        gene_j = pathway_genes[j]
        
        # Identify cell lines where BOTH genes have non-zero excess mutations
        co_occurrence_mask = (sub_matrix[:, i] > 0.0) & (sub_matrix[:, j] > 0.0)
        co_occurring_cells_count = np.sum(co_occurrence_mask)
        
        if co_occurring_cells_count == 0:
            continue # No edge if they never mutate together in the leading edge
            
        # Extract the sum of the pair's scores for co-occurring cells
        pair_sum = sub_matrix[co_occurrence_mask, i] + sub_matrix[co_occurrence_mask, j]
        # Extract the corresponding cell pathway totals
        pathway_total_subset = cell_pathway_totals[co_occurrence_mask, 0]
        
        # Calculate the proportional weight per cell, then average them
        relative_dominance_per_cell = pair_sum / pathway_total_subset
        avg_edge_weight = np.mean(relative_dominance_per_cell)
        
        edge_accumulator.append({
            'Source': gene_i,
            'Target': gene_j,
            'Edge_Weight': avg_edge_weight,
            'Co_Occurrence_Count': int(co_occurring_cells_count),
            'Co_Occurrence_Prevalence': co_occurring_cells_count / n_cells
        })
        
    edges_df = pd.DataFrame(edge_accumulator)
    return nodes_df, edges_df
