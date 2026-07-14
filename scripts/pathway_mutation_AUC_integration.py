import pandas as pd
import bioframe as bf
import numpy as np
import matplotlib.pyplot as plt
import xlmhglite
import networkx as nx
from pathlib import Path
from scipy.stats import beta, false_discovery_control

from src.utils.io import get_vcf_summary_tbl, extract_gene_anno, parse_gmt
from src.mutation.gene_set_analysis import *
from src.dose_response.detect_response import *
# %%

vcf_folder = Path("/home/vipink/Documents/dose_response_workflow/data/omics/mutations_wes_vcf_20250226/")
vcf_file_list = list(vcf_folder.glob("*.gz"))
gene_set_file = "/home/vipink/Documents/dose_response_workflow/data/metadata/c6.all.v2026.1.Hs.symbols.gmt"

dose_fit_file = "/home/vipink/Documents/dose_response_workflow/data/GDSC2_fitted_dose_response_27Oct23.csv"
# %%

all_wes_mutation_df = pd.concat([get_vcf_summary_tbl(vcf_file) for vcf_file in vcf_file_list]).drop_duplicates()


gene_set_dict = parse_gmt(gene_set_file)
Gene_Set_size_tbl = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).explode('Genes').Gene_Set.value_counts().reset_index().rename(columns={'count':'gene_count'})

Gene_Set_avg_gene_in_count_df = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).explode('Genes').merge(all_wes_mutation_df.loc[:,['sanger_model_id','gene']],how='left',left_on='Genes',right_on='gene').assign(out = lambda df: ~df.gene.isna()).groupby(['Gene_Set','sanger_model_id']).agg(gene_in = ('out','sum')).reset_index()

# %%

Gene_Set_tot_count_tbl = Gene_Set_avg_gene_in_count_df.groupby('Gene_Set').agg(tot_count = ('gene_in','sum')).reset_index()
cell_id_number = Gene_Set_avg_gene_in_count_df.sanger_model_id.nunique() 
total_zscore_tbl = parallel_zscore_estimation(Gene_Set_avg_gene_in_count_df,Gene_Set_tot_count_tbl,cell_id_number, n_cores=10)

# %%

dose_coef_tbl = pd.read_csv(dose_fit_file,sep='\t')
dose_data_tbl = dose_coef_tbl.loc[:,['SANGER_MODEL_ID','CANCER_TYPE', 'DRUG_ID', 'DRUG_NAME','MIN_CONC', 'MAX_CONC','LN_IC50','AUC', 'RMSE']].query('RMSE < 0.2')

# %%

null_dose_data_tbl = dose_data_tbl.assign(LN_MAX_CONC = lambda df: np.log(df.MAX_CONC)).assign(inert = lambda df: df.LN_IC50.gt(df.LN_MAX_CONC)).query('inert')

# %%

drug_beta_param_df = get_shrunk_beta_params(null_dose_data_tbl)

# %%

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
tmp_ax = total_zscore_tbl.query('Gene_Set == "RAF_UP.V1_UP"').merge(dose_data_tbl.loc[:,['SANGER_MODEL_ID','DRUG_ID','sensitivity_p']].query('DRUG_ID == 1561'),left_on='cell_id',right_on='SANGER_MODEL_ID').assign(zrank = lambda df: df.z_score.rank(pct=True),psrank = lambda df: df.sensitivity_p.rank(pct=True)).plot.scatter(x='psrank',y = 'sensitivity_p')
plt.show()

# %%

tmp_drug = 1561

gene_set_drug_pair_list = [(path, tmp_drug) for path in total_zscore_tbl.Gene_Set.unique()]

d_res_df = parallel_mhgt(gene_set_drug_pair_list,dose_data_tbl,total_zscore_tbl,n_cores=10)
# %%

tmp_ax = d_res_df.assign(fdr = lambda df: false_discovery_control(df.pvalue)).fdr.plot.kde(title = 'FDR corrected p-vaue')
plt.show()

# %%
tmp_path = "YAP1_UP"

gene_set_drug_pair_list = [(tmp_path, drug) for drug in dose_data_tbl.DRUG_ID.unique()]

p_res_df = parallel_mhgt(gene_set_drug_pair_list,dose_data_tbl,total_zscore_tbl,n_cores=10)

# %%

tmp_ax = p_res_df.assign(fdr = lambda df: false_discovery_control(df.pvalue)).fdr.plot.kde(title = 'FDR corrected p-vaue')
plt.show()

# %%

# How choice of threshold impacts significance of association
tmp_drug = 1051
#tmp_drug = 1821


tmp_zarray = total_zscore_tbl.query('Gene_Set == @tmp_path').query('z_score > 0').z_score.to_numpy()


z_tresh_array = np.arange(tmp_zarray.min(),tmp_zarray.max(),0.05)

tmp_res_tbl = pd.concat([get_hgmt_result_tbl(tmp_path,tmp_drug,total_zscore_tbl,dose_data_tbl,tmp_z) for tmp_z in z_tresh_array]).assign(fdr = lambda df: false_discovery_control(df.pvalue))

tmp_ax = tmp_res_tbl.plot(x='z_thresh',y='pvalue',logy=True,title = f"{tmp_path}",xlabel = "z-score threshold", ylabel="drug sensitivity association p-value")
plt.show()
