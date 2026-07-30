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
drug_id = 1010
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

# %%

tmp_drug_out_path_excess_count_df = gene_set_collection_excess_count_df.loc[:,out_path]

out_path_leading_edge_member_list = tmp_res.query('Pathway_Name in @out_path').loc[:,['Pathway_Name','Leading_Edge_Cell_Lines']].set_index('Pathway_Name').loc[tmp_drug_out_path_excess_count_df.columns,'Leading_Edge_Cell_Lines'].to_list()

def compute_leading_edge_quantiles_vectorized(
    burdens_df: pd.DataFrame, 
    le_cell_lists: list[list[str]]
) -> pd.DataFrame:
    """
    Computes F_{c,p} (the LE quantile score) for all cell lines across all pathways.
    
    Parameters:
    -----------
    burdens_df : pd.DataFrame (N cell lines x K pathways)
        Matrix of continuous pathway mutation burdens.
    le_cell_lists : list of list of str (length K)
        List containing the list of LE cell line identifiers for each pathway.
        Order must match burdens_df.columns.
    Returns:
    --------
    F_df : pd.DataFrame (N cell lines x K pathways)
        Matrix of continuous LE depth scores bounded in [0, 1].
    """
    cell_ids = burdens_df.index
    pathway_names = burdens_df.columns
    N, K = burdens_df.shape
    
    # 1. Convert burdens to 2D NumPy array (N x K)
    B = burdens_df.values
    
    # 2. Build Binary LE Indicator Matrix M (N x K)
    # M[i, j] = 1 if cell line i is in LE_j, else 0
    cell_to_idx = {cell: i for i, cell in enumerate(cell_ids)}
    M = np.zeros((N, K), dtype=bool)
    
    for p_idx, le_cells in enumerate(le_cell_lists):
        valid_indices = [cell_to_idx[c] for c in le_cells if c in cell_to_idx]
        M[valid_indices, p_idx] = True
    # 3. Compute Size of LE per pathway (1 x K)
    le_counts = M.sum(axis=0, keepdims=True) # shape: (1, K)
    # Avoid division by zero if a pathway has 0 LE cells
    le_counts_safe = np.where(le_counts == 0, 1, le_counts)
    # 4. FULLY VECTORIZED ECDF COMPUTATION
    # --------------------------------------------------------------------------
    # For memory efficiency on large matrices, we operate column-by-column across 
    # pathways (K iterations), broadcasting N x N comparisons in vectorized C memory.
    F = np.zeros((N, K), dtype=float)
    
    for p in range(K):
        if le_counts[0, p] == 0:
            continue
            
        # Extract all cell line burdens for pathway p: shape (N, 1)
        b_all = B[:, p:p+1]
        
        # Extract ONLY leading-edge burdens for pathway p: shape (1, n_p)
        b_le = B[M[:, p], p:p+1].T
        
        # Outer comparison matrix via broadcasting: shape (N, n_p)
        # Compares every cell line burden against all LE burdens for this pathway
        comparison_matrix = (b_le <= b_all)
        
        # Sum across LE cells and divide by total LE size: shape (N,)
        F[:, p] = comparison_matrix.mean(axis=1)
    # 5. Zero out cell lines that do not meet the minimum LE threshold cutoff T_p*
    # Optional guard: if a cell's burden is lower than min(LE_p), ensure score is 0.0
    # min_le_burden = np.where(M, B, np.inf).min(axis=0, keepdims=True)
    # F = np.where(B >= min_le_burden, F, 0.0)
    return pd.DataFrame(F, index=cell_ids, columns=pathway_names)

# %%

out_path_leading_edge_score_tbl = compute_leading_edge_quantiles_vectorized(tmp_drug_out_path_excess_count_df,out_path_leading_edge_member_list)
# %%
tmp_ax = (
        out_path_leading_edge_score_tbl.sum(axis=1).sort_values().reset_index().rename(columns= {0:'score'}).merge(
dose_data_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID','AUC','sensitivity_p']],
left_on='sanger_model_id',
right_on='SANGER_MODEL_ID',
how='left'
        )
.dropna()
.plot
.scatter(x='score',y='AUC')
)
plt.show()

# %%

def augment_features_for_or_logic(F_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments the leading-edge feature matrix F with max-pooling 
    and hit-count features to help linear models capture OR-gate logic.
    """
    F_augmented = F_df.copy()
    
    # 1. Max-Pooling Feature: Captures the single strongest pathway hit (OR-gate helper)
    F_augmented['F_max'] = F_df.max(axis=1)
    
    # 2. Hit Count / MPV-Score: Captures cumulative pathway exceedances (> 0 threshold)
    F_augmented['F_count'] = (F_df > 0).sum(axis=1)
    
    return F_augmented

# %%
cell_ids = tmp_drug_out_path_excess_count_df.index
pathway_names = tmp_drug_out_path_excess_count_df.columns
all_le_cells_set = set(c for sublist in out_path_leading_edge_member_list for c in sublist)
y_union = pd.Series(cell_ids.isin(all_le_cells_set).astype(int), index=cell_ids)

F_augmented_df = augment_features_for_or_logic(out_path_leading_edge_score_tbl)
# -------------------------------------------------------------------------
# 3. FIT LOGISTIC REGRESSION ON FULL COHORT
# -------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
clf = LogisticRegression(
    l1_ratio=0,
    solver='liblinear',
    random_state=42
)

clf.fit(out_path_leading_edge_score_tbl, y_union)
y_probs = clf.predict_proba(out_path_leading_edge_score_tbl)[:, 1]


clf.fit(F_augmented_df, y_union)
y_probs = clf.predict_proba(F_augmented_df)[:, 1]
# %%
# -------------------------------------------------------------------------
# 4. EVALUATE PROOF-OF-CONCEPT METRICS
# -------------------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_union, y_probs)
pr_auc = auc(recall, precision)
roc_auc = roc_auc_score(y_union, y_probs)


print("\n--- PoC Model Performance ---")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")

# %%
# Extract non-zero learned coefficients (Selected Pathways)
coef_df = pd.DataFrame({
    'Pathway': F_augmented_df.columns,
    'Coefficient (Beta)': clf.coef_[0],
    'Odds_Ratio': np.exp(clf.coef_[0])
}).sort_values(by='Coefficient (Beta)', ascending=False)

selected_coefs = coef_df[coef_df['Coefficient (Beta)'] > 0]

# %%

import matplotlib.pyplot as plt
import seaborn as sns

# Compare F_count vs F_max colored by LE_union target
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=F_augmented_df, 
    x='F_count', 
    y='F_max', 
    hue=y_union, 
    alpha=0.7,
    palette={0: 'gray', 1: 'red'}
)
plt.title("Distribution of LE_union Responders across F_count and F_max")
plt.xlabel("Pathway Hit Count (F_count)")
plt.ylabel("Max Pathway Depth (F_max)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
