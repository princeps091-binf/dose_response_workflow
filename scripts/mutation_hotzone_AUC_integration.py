import pandas as pd
import bioframe as bf
import numpy as np
import matplotlib.pyplot as plt
import xlmhglite
import networkx as nx
import fast_hdbscan
from pathlib import Path

# %%
from src.utils.io import import_vcf_as_df
from src.utils.hdbscan_processing import *
# %%
vcf_folder = Path("/home/vipink/Documents/dose_response_workflow/data/omics/mutations_wgs_vcf_20260305/")

vcf_file_list = list(vcf_folder.glob("*.gz"))

dose_fit_file = "/home/vipink/Documents/dose_response_workflow/data/GDSC2_fitted_dose_response_27Oct23.csv"
# %%

all_wgs_mutations_df = pd.concat([import_vcf_as_df(f).loc[:,['#CHROM','POS']].assign(vcf= f.stem.split('_')[0]) for f in vcf_file_list]).rename(columns={'#CHROM':'chrom'})
# %%

tmp_chrom = 'chr19'
chrom_wgs_mutation_df = all_wgs_mutations_df.query('chrom == @tmp_chrom')
# %%
chrom_pos_summary_tbl = chrom_wgs_mutation_df.value_counts().reset_index().groupby(['chrom','POS']).agg(avg_injury=('count','mean')).reset_index().merge(chrom_wgs_mutation_df.groupby(['chrom','POS']).agg(nsample=('vcf','nunique')).reset_index()).sort_values('nsample')
# %%

clusterer = fast_hdbscan.fast_hdbscan(chrom_pos_summary_tbl.loc[:,['POS']],sample_weights=chrom_pos_summary_tbl.nsample.to_numpy().astype(np.float32),min_cluster_size=2, return_trees= True)
chrom_pos_summary_tbl = chrom_pos_summary_tbl.assign(hdbscan_label = clusterer[0])
# %%
chrom_cluster_coord_tbl = chrom_pos_summary_tbl.groupby('hdbscan_label').agg(chrom = ('chrom','first'),start = ('POS','min'),end = ('POS','max'),n = ('POS','nunique')).reset_index().query('hdbscan_label >= 0').loc[:,['chrom','start','end','n','hdbscan_label']]
# %%
root = chrom_pos_summary_tbl.shape[0]

tree_edge_df = pd.DataFrame({'parent':clusterer[3].parent,'child':clusterer[3].child,'lambda':clusterer[3].lambda_val})

tree_nx = nx.from_pandas_edgelist(
    tree_edge_df, 
    source='parent', 
    target='child', 
    edge_attr=True,     # Keeps other columns as edge attributes
    create_using=nx.DiGraph()
)
# %%
hdb_cluster_df = produce_hdb_cluster_summary_tbl(tree_edge_df,tree_nx,chrom_pos_summary_tbl,root)
# %%

bf.overlap(hdb_cluster_df.loc[:,['start','end','n_event']].assign(chrom = tmp_chrom,end = lambda df: df.end + 1).loc[:,['chrom','start','end','n_event']],chrom_wgs_mutation_df.rename(columns={'POS':'start'}).assign(end = lambda df: df.start + 1).loc[:,['chrom','start','end','vcf']]).groupby(['chrom','start','end','n_event']).agg(nsample = ('vcf_','unique'))
# %%
dose_coef_tbl = pd.read_csv(dose_fit_file,sep='\t')
dose_data_tbl = dose_coef_tbl.loc[:,['SANGER_MODEL_ID','CANCER_TYPE', 'DRUG_ID', 'DRUG_NAME','MIN_CONC', 'MAX_CONC','LN_IC50','AUC', 'RMSE']]
# %%
# Mild dependency on top AUC with dosage span
def get_q(x_df):
    return x_df.quantile(0.01)

tmp_ax = dose_data_tbl.assign(CONC_span = lambda df: np.log10(df.MAX_CONC - df.MIN_CONC),LN_MAX_CONC = lambda df: np.log(df.MAX_CONC)).query('LN_IC50 > LN_MAX_CONC').groupby('CONC_span').agg(qAUC = ('AUC',get_q)).reset_index().plot.scatter(x= 'CONC_span',y='qAUC')
plt.show()
# %%
# When LN_IC50 > MAX_CONC often poor AUC -> proxy for unresponsive ?

tmp_ax = dose_data_tbl.assign(LN_MAX_CONC = lambda df: np.log(df.MAX_CONC)).assign(responsive = lambda df: df.LN_IC50.gt(df.LN_MAX_CONC)).query('DRUG_ID == 2175').groupby('responsive').AUC.plot.kde(legend=True)
plt.show()

# %%

null_dose_data_tbl = dose_data_tbl.assign(LN_MAX_CONC = lambda df: np.log(df.MAX_CONC)).assign(inert = lambda df: df.LN_IC50.gt(df.LN_MAX_CONC)).query('inert')
# %%
from scipy.stats import beta, gaussian_kde
# 2. Setup Plotting Range
x = np.linspace(0, 1.05, 500)

plt.figure(figsize=(9, 5))

# 3. Calculate KDE manually
# gaussian_kde creates a continuous function from discrete data points
for drug_id in null_dose_data_tbl.DRUG_ID.unique():
    raw_data = null_dose_data_tbl.query('DRUG_ID == @drug_id').AUC.to_numpy()
    kde_function = gaussian_kde(raw_data)
    kde_values = kde_function(x)
    # 5. Plotting
    # Observed KDE (Dashed Line)
    plt.plot(x, kde_values, color='black', alpha=0.15,linewidth=1)

# Formatting
plt.title(f"AUC null values per drug", fontsize=14)
plt.xlabel("AUC", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(0.5, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# %%


def get_shrunk_beta_params(df_null):
    # 1. Global Stats
    global_mean = df_null['AUC'].mean()
    global_var = df_null['AUC'].var()
    
    # 2. Per-Drug Stats
    stats = df_null.groupby('DRUG_ID')['AUC'].agg(['mean', 'var', 'count'])
    
    # 3. Calculate Shrinkage Factor (w)
    # Higher count = w closer to 1 (trust the drug data)
    # Lower count = w closer to 0 (trust the global mean)
    # This formula is a frequentist approximation of a random effect
    w = stats['count'] / (stats['count'] + (global_var / stats['var'].fillna(global_var)))
    
    stats['shrunk_mean'] = w * stats['mean'] + (1 - w) * global_mean
    
    # 4. Convert Shrunk Mean back to Beta parameters (alpha, beta)
    # We assume a constant precision (phi) for the whole assay
    # phi = (mean * (1 - mean) / var) - 1
    phi_global = (global_mean * (1 - global_mean) / global_var) - 1
    phi = max(phi_global, 0.01)
    stats['alpha'] = stats['shrunk_mean'] * phi
    stats['beta'] = (1 - stats['shrunk_mean']) * phi
    
    return stats.reset_index().loc[:,['DRUG_ID','alpha', 'beta']]

from scipy.stats import beta
# %%
drug_beta_param_df = get_shrunk_beta_params(null_dose_data_tbl)
# %%
plt.figure(figsize=(12, 6))

# 1. Define the AUC range (usually focusing on the 0.5 - 1.0 range)
x = np.linspace(0, 1, 1000)

# 2. Plot each drug as a faint curve
for _, row in drug_beta_param_df.iterrows():
    y = beta.pdf(x, row['alpha'], row['beta'])
    plt.plot(x, y, color='steelblue', alpha=0.15, linewidth=1)

# 3. Plot the 'Population Average' for context
# This represents the center of the 'Chemical Universe'
mean_a = drug_beta_param_df['alpha'].mean()
mean_b = drug_beta_param_df['beta'].mean()
y_avg = beta.pdf(x, mean_a, mean_b)
plt.plot(x, y_avg, color='firebrick', linewidth=3, label='Global Null (Population Average)')

plt.title("The 'Inert Null' Landscape Across All Drugs", fontsize=15)
plt.xlabel("AUC (Area Under the Curve)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.xlim(0.6, 1.05) # Focus on the high-AUC region where nulls live
plt.grid(axis='y', alpha=0.3)

plt.show()

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
tmp_ax = dose_data_tbl.plot.scatter(x='AUC',y='adjusted_auc_rank',c='alpha',s=1,alpha=0.2,logy=True)
plt.show()
# %%

bf.overlap(hdb_cluster_df.assign(chrom = tmp_chrom,end = lambda df: df.end + 1).loc[:,['chrom','start','end','n_event','hdb_cluster']],chrom_wgs_mutation_df.rename(columns={'POS':'start'}).assign(end = lambda df: df.start + 1).loc[:,['chrom','start','end','vcf']])
# %%
tmp_cl = 238138
tmp_cl_df = hdb_cluster_df.query('hdb_cluster == @tmp_cl')
tmp_cl_vcf_array = bf.overlap(tmp_cl_df.assign(chrom = tmp_chrom,end = lambda df: df.end + 1).loc[:,['chrom','start','end','n_event','hdb_cluster']],chrom_wgs_mutation_df.rename(columns={'POS':'start'}).assign(end = lambda df: df.start + 1).loc[:,['chrom','start','end','vcf']]).vcf_.unique()
cl_ranked_array = dose_data_tbl.query('DRUG_ID == 1003').assign(in_cl = lambda df: np.where(df.SANGER_MODEL_ID.isin(tmp_cl_vcf_array),1,0)).sort_values('sensitivity_p').in_cl.to_numpy()
from multiprocessing import Pool
# 1. We define a global 'initializer' to make the big matrix accessible
# to all workers without passing it as a repeated argument.
def init_worker(shared_count_tbl,gene_set_tot_count_tbl,n_cell_id):
    global global_count_tbl, global_sums, global_n
    global_count_tbl = shared_count_tbl
    global_sums = gene_set_tot_count_tbl
    global_n = n_cell_id
    

def _get_cell_line_pathway_mutation_enrichment_tbl_p(cell_line_id):
    
#    tmp_cell_model_count_tbl = global_count_tbl.query('sanger_model_id == @cell_line_id').loc[:,['Gene_Set','gene_in']].merge(global_count_tbl.query('sanger_model_id != @cell_line_id').loc[:,['Gene_Set','gene_in']].groupby(['Gene_Set']).agg(avg_gene = ('gene_in','mean')).reset_index())

    tmp_cell_model_count_tbl = global_count_tbl.query('sanger_model_id == @cell_line_id').loc[:,['Gene_Set','gene_in']]
    tmp_cell_model_count_tbl = tmp_cell_model_count_tbl.merge(global_sums, on='Gene_Set').assign(avg_gene = lambda df: (df.tot_count - df.gene_in)/ (global_n - 1))
    
    tmp_cell_model_count_tbl = get_mutation_burden_zscore(tmp_cell_model_count_tbl)
    
    return tmp_cell_model_count_tbl.assign(cell_id = cell_line_id)

def parallel_zscore_estimation(Gene_Set_count_tbl,Gene_Set_tot_count_tbl, n_cell_id, n_cores=10):
    # Create the Pool
    # 'initializer' runs once for every worker when they start
    with Pool(processes=n_cores, 
              initializer=init_worker, 
              initargs=(Gene_Set_count_tbl,Gene_Set_tot_count_tbl,n_cell_id)) as pool:
        
        # We pass the integer indices of the rows
        cell_id_list = Gene_Set_count_tbl.sanger_model_id.unique() 
        
        # map() will distribute the work and collect results in order
        results = pool.map(_get_cell_line_pathway_mutation_enrichment_tbl_p, cell_id_list)
        
    # Reconstruct the DataFrame
    zscore_df = pd.concat(results)
    return zscore_df

xlmhglite.xlmhg_test(cl_ranked_array, X=2, L=len(cl_ranked_array))

