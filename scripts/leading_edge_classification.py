import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import reload
from scipy.stats import beta, hypergeom
import networkx as nx
import itertools
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
# Paclitaxel = 1080
# Vemurafenib = 
# Gefitinib = 1010
#1 Trametinib = 1372 
# Osimertinib = 1919 ->
drug_id = 1919
drug_name = dose_data_tbl.query('DRUG_ID == @drug_id').DRUG_NAME.iloc[0]
tmp_drug_excess_mutation_count_tbl = src.mutation.gene_set_analysis.get_excess_mutation_count_matrix(drug_id,K_matrix,N_vector,dose_data_tbl,all_cells,all_genes)


gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_excess_mutation_count_tbl,all_cells,all_genes,gene_set_to_use_dict)


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

tmp_ax = kneed_tbl.plot(x='x',y='y')
plt.show()

direction, curve = find_shape(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy())
kl = KneeLocator(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy(), curve=curve, direction=direction)

# %%
tot_cell_lines = np.zeros(kneed_tbl.x.nunique())
for tmp_idx, tmp_x in enumerate(kneed_tbl.x.unique()):
    out_path = kneed_tbl.query('x <= @tmp_x').Pathway_Name.to_list()
    tmp_drug_all_leading_edge_cell_lines_list = tmp_res.query('Pathway_Name in @out_path').Leading_Edge_Cell_Lines.explode().unique()
    tot_cell_lines[tmp_idx] = len(tmp_drug_all_leading_edge_cell_lines_list)

# %%

tmp_ax = pd.DataFrame({'x':kneed_tbl.x.unique(),'ncell':tot_cell_lines}).plot(x='x',y='ncell',logx=True)

tmp_ax.axvline(x=kl.knee, color='red', linestyle='--', linewidth=2, label='Threshold')
plt.show()

# %%
tmp_ax = dose_data_tbl.query('DRUG_ID == @drug_id').assign(leading_edge = lambda df: df.SANGER_MODEL_ID.isin(tmp_drug_all_leading_edge_cell_lines_list)).groupby('leading_edge').AUC.plot.kde(legend=True)

plt.show()

# %%

out_path = kneed_tbl.query('x <= @kl.knee').Pathway_Name.to_list()
# %%


drug_to_cell_assoc_tbl = tmp_res.loc[:,['Pathway_Name','Leading_Edge_Cell_Lines','Min_mHG_P_Value']].assign(mHG_q = lambda df: pd.qcut(df.Min_mHG_P_Value,q=50)).explode('Leading_Edge_Cell_Lines').merge(dose_data_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID','sensitivity_p','AUC']],left_on='Leading_Edge_Cell_Lines',right_on='SANGER_MODEL_ID',how='left')

drug_to_cell_assoc_tbl.groupby(['SANGER_MODEL_ID','AUC','sensitivity_p']).agg(min_mHG =('Min_mHG_P_Value','min'),max_mHG = ('Min_mHG_P_Value','max') ).sort_values('sensitivity_p').reset_index()

tmp_LE_list = tmp_res.query('Pathway_Name == @out_path[0]').Leading_Edge_Cell_Lines.to_list()[0]
gene_set_collection_excess_count_df.loc[tmp_LE_list,out_path[0]]

gene_set_collection_excess_count_df.loc['SIDM00210',out_path[0]]

tmp_drug_out_path_thresh = tmp_res.query('Pathway_Name in @out_path').loc[:,['Pathway_Name','Optimal_Burden_Threshold_Tau']]
tmp_drug_out_path_excess_count_df = gene_set_collection_excess_count_df.loc[:,out_path]

tmp_drug_out_path_excess_count_df.gt(tmp_drug_out_path_thresh.set_index('Pathway_Name').loc[tmp_drug_out_path_excess_count_df.columns.to_list(),'Optimal_Burden_Threshold_Tau'].to_numpy(),axis=1).sum(axis=1).sort_values()


