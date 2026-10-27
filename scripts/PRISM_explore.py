# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import reload
from scipy.stats import beta, hypergeom
import src.utils.io 
import src.mutation.gene_set_analysis
import src.dose_response.detect_response
import src.integration.gene_burden
import src.integration.leading_edge

reload(src.integration.gene_burden)
reload(src.mutation.gene_set_analysis)
reload(src.dose_response.detect_response)
reload(src.utils.io)
reload(src.integration.leading_edge)
# %%

dose_tbl = pd.read_csv('./data/DepMap/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv')

mutation_tbl = pd.read_csv('./data/DepMap/OmicsSomaticMutations.csv',sep=',')

# %%

gene_set_file = "/home/vipink/Documents/dose_response_workflow/data/metadata/c2.all.v2026.1.Hs.symbols.gmt"

gene_set_dict = src.utils.io.parse_gmt(gene_set_file)
Gene_Set_size_tbl = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).explode('Genes').Gene_Set.value_counts().reset_index().rename(columns={'count':'gene_count'})

sub_collection_list = ['REACTOME','KEGG','PID','WP']
collection_to_use_list = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).assign(collection = lambda df: [i.split('_')[0] for i in df.Gene_Set]).query('collection in @sub_collection_list').Gene_Set.drop_duplicates().to_list()

gene_set_to_use_dict = {k: gene_set_dict[k] for k in collection_to_use_list if k in gene_set_dict}

# %%


dose_data_tbl = dose_tbl.query('screen_id == "MTS010"').loc[:,['depmap_id','ccle_name', 'name', 'lower_limit','upper_limit','ic50','auc', 'r2']].query('r2 > 0')


null_dose_data_tbl = dose_data_tbl.assign(inert = lambda df: df.ic50.gt(df.upper_limit)).query('inert')

drug_beta_param_df = src.dose_response.detect_response.get_shrunk_beta_params(null_dose_data_tbl,value_col='auc',id_col='name')

dose_data_tbl = dose_data_tbl.merge(
        drug_beta_param_df[['name','alpha', 'beta']], 
        how='left')
# 2. Vectorized CDF calculation
# This gives the probability of observing 'auc' or lower given the Null
dose_data_tbl = dose_data_tbl.assign(sensitivity_p = lambda df: beta.cdf(df['auc'], df['alpha'], df['beta']))
# 3. For ranking, we often use the Negative Log 10 of the probability
# This makes 'stronger' hits have higher positive values
dose_data_tbl = dose_data_tbl.dropna().assign(rank_score = lambda df: -np.log10(df.sensitivity_p + 1e-100)) # Avoid log(0)



# %%
# Calculate fixed total exome mutation burden per cell line (Trials)
print("Step 1: Pivoting mutation counts and building core exome vectors...")
total_exome_loads = mutation_tbl.groupby('ModelID').size().rename('total_cell_muts')

base_matrix = mutation_tbl.groupby(['ModelID', 'HugoSymbol']).size().unstack(fill_value=0)

all_cells = base_matrix.index.tolist()
all_genes = base_matrix.columns.tolist()

# Core numpy structures
K_matrix = base_matrix.values  # Shape: (n_cells, n_genes)
N_vector = total_exome_loads.reindex(all_cells).fillna(0).values.reshape(-1, 1) # Shape: (n_cells, 1)

# %%

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

# %%

#1 Trametinib = 1372 
#2 Paclitaxel = 1080
#3 Vorinostat = 1012
drug_id = 'osimertinib'
tmp_drug_resistant_cell_lines = null_dose_data_tbl.query('name == @drug_id').depmap_id.tolist()
dose_data_tbl = dose_data_tbl.assign(is_resistant = lambda df: df.depmap_id.isin(tmp_drug_resistant_cell_lines))
tmp_drug_excess_mutation_count_tbl = src.mutation.gene_set_analysis.get_excess_mutation_count_matrix(drug_id,K_matrix,N_vector,dose_data_tbl,all_cells,all_genes,group_id='name',cell_id='depmap_id',gene_name='HugoSymbol')

tmp_drug_data_tbl = dose_data_tbl.query('name == @drug_id')

gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_excess_mutation_count_tbl,all_cells,all_genes,gene_set_to_use_dict,cell_id='depmap_id',gene_name='HugoSymbol')

shared_cell_id_list = list(set(tmp_drug_data_tbl.depmap_id.unique().tolist()).intersection(gene_set_collection_excess_count_df.index.tolist())) 

tmp_drug_sensitivity_series = tmp_drug_data_tbl.query('depmap_id in @shared_cell_id_list').set_index('depmap_id').loc[shared_cell_id_list,'sensitivity_p']

tmp_drug_excess_mutation_count_tbl = gene_set_collection_excess_count_df.loc[shared_cell_id_list,:]
# %%

tmp_drug_sensitivity_series = tmp_drug_data_tbl.query('depmap_id in @tmp_drug_data_tbl.depmap_id.tolist()').set_index('depmap_id').loc[shared_cell_id_list,'sensitivity_p']
tmp_res = src.mutation.gene_set_analysis.run_high_throughput_parallel_xlmhg(
    pathway_burden_df = tmp_drug_excess_mutation_count_tbl,
    drug_sensitivity_df = tmp_drug_sensitivity_series.reset_index(), 
    n_burden_steps = 20,
    auc_col = 'sensitivity_p',
    sanger_id_col = 'depmap_id',
    n_jobs = 8  
   )
