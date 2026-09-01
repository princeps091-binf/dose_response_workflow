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

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

reload(src.integration.gene_burden)
reload(src.mutation.gene_set_analysis)
reload(src.dose_response.detect_response)
reload(src.utils.io)
reload(src.integration.leading_edge)

# %%


class OutOfFoldEnsemble:
    """Wraps K fold models to predict averaged out-of-fold probabilities on new data."""
    def __init__(self, models, optuna_params, coef_df):
        self.models = models
        self.optuna = optuna_params
        self.coef_df = coef_df
    def predict_proba(self, X):
        # Gather predicted positive-class probabilities from each fold model
        fold_probs = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models])
        # Average across all K models
        mean_probs = np.mean(fold_probs, axis=1)
        # Return standard 2D probability array [P(y=0), P(y=1)]
        return np.column_stack([1 - mean_probs, mean_probs])
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)




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
drug_id = 1372
drug_name = dose_data_tbl.query('DRUG_ID == @drug_id').DRUG_NAME.iloc[0]
tmp_drug_excess_mutation_count_tbl = src.mutation.gene_set_analysis.get_excess_mutation_count_matrix(drug_id,K_matrix,N_vector,dose_data_tbl,all_cells,all_genes)


gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_excess_mutation_count_tbl,all_cells,all_genes,gene_set_to_use_dict)


tmp_res = src.mutation.gene_set_analysis.run_high_throughput_parallel_xlmhg(
    pathway_burden_df = gene_set_collection_excess_count_df,   
    drug_sensitivity_df = dose_data_tbl.query('DRUG_ID == @drug_id'), 
    n_burden_steps = 20,
    auc_col = 'sensitivity_p',
    sanger_id_col = 'SANGER_MODEL_ID',
    n_jobs = 8  
)

tmp_res = tmp_res.assign(x = lambda df: df.Min_mHG_P_Value.rank(pct=True),y=lambda df:-np.log10(df.Min_mHG_P_Value)).sort_values('x')


# %%
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
import joblib
best_ensemble_path = f"./data/tmp_res/logit_ensemble_drug_{drug_id}.joblib"

best_ensemble = joblib.load(best_ensemble_path)


# %%
leading_edge_member_list = tmp_res.loc[:,['Pathway_Name','Leading_Edge_Cell_Lines']].set_index('Pathway_Name').loc[gene_set_collection_excess_count_df.columns,'Leading_Edge_Cell_Lines'].to_list()



leading_edge_score_tbl = compute_leading_edge_quantiles_vectorized(gene_set_collection_excess_count_df,leading_edge_member_list)

# %%
def augment_features_for_or_logic(F_df: pd.DataFrame, pathway_rank_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments the leading-edge feature matrix F with max-pooling 
    and hit-count features to help linear models capture OR-gate logic.
    """
    common_pathways = F_df.columns.intersection(pathway_rank_df.Pathway_Name)
    
    if len(common_pathways) == 0:
        raise ValueError("No matching pathways found between F_matrix columns and pathway_mhg_pvalues index.")
    F_augmented = F_df.copy().loc[:,common_pathways]
    pathway_ranking = 1 - pathway_rank_df.loc[:,['Pathway_Name','x']].set_index('Pathway_Name').loc[common_pathways].assign(tmp_rank = lambda df: df.x.rank(pct=True,ascending=True))
    # 1. Max-Pooling Feature: Captures the single strongest pathway hit (OR-gate helper)
    lead_max = F_df.max(axis=1)
    top_pathway_per_cell = F_df.idxmax(axis=1)
    cell_top_ranks = pathway_ranking.loc[top_pathway_per_cell]
    F_augmented['F_max_rank'] = cell_top_ranks.tmp_rank.to_numpy()
    F_augmented['F_max'] = lead_max
    F_augmented['F_max_weight'] = F_augmented['F_max'] * F_augmented['F_max_rank']
    # 2. Hit Count / MPV-Score: Captures cumulative pathway exceedances (> 0 threshold)
    F_augmented['F_count'] = (F_df > 0).sum(axis=1)
    return F_augmented

# %%
tmp_thresh = best_ensemble.optuna['tmp_thresh']
out_path = tmp_res.query('x <= @tmp_thresh').Pathway_Name.to_list()
tmp_drug_out_path_excess_count_df = gene_set_collection_excess_count_df.loc[:, out_path]
out_path_leading_edge_score_tbl = leading_edge_score_tbl.loc[:, out_path]
out_path_leading_edge_member_list = (
    tmp_res.query('Pathway_Name in @out_path')
    .Leading_Edge_Cell_Lines.explode()
    .unique()
    .tolist()
)

cell_ids = tmp_drug_out_path_excess_count_df.index

LE_count_tbl = (
    tmp_res.query('Pathway_Name in @out_path')
    .Leading_Edge_Cell_Lines.explode()
    .value_counts()
    .reset_index()
    .rename(columns={'count': 'path_count'})
)

LE_cells = LE_count_tbl.query('path_count > 0').Leading_Edge_Cell_Lines.to_list()
y_union = pd.Series(cell_ids.isin(LE_cells).astype(int), index=cell_ids)

F_augmented_df = augment_features_for_or_logic(out_path_leading_edge_score_tbl, tmp_res)


new_predictions = best_ensemble.predict_proba(F_augmented_df)[:, 1]

optim_pred_df = pd.DataFrame({'proba':new_predictions,'sanger_model_id':F_augmented_df.index.tolist()}).merge(y_union.reset_index()).rename(columns={0:'LE'})

tmp_ax = optim_pred_df.plot.scatter(x='proba',y='LE',alpha=0.1)
plt.show()

# %%
credible_LE_cells_list = optim_pred_df.query('proba > 0.5 and LE ==1').sanger_model_id.unique()


summary_coefs = pd.DataFrame({
    'mean_coef': best_ensemble.coef_df.mean(axis=0),
    'min_coef': best_ensemble.coef_df.min(axis=0),
    'max_coef': best_ensemble.coef_df.max(axis=0),
    'selection_freq': (best_ensemble.coef_df != 0).mean(axis=0)
}).sort_values(by='selection_freq', ascending=False)
model_path_to_keep_list = summary_coefs.query('selection_freq > 0.5').index.tolist()
# %%
gene_in_optim_model_list = list(set().union(*(gene_set_to_use_dict[k] for k in model_path_to_keep_list if k in gene_set_to_use_dict)))

model_gene_set_dict = {k: gene_set_to_use_dict[k] for k in model_path_to_keep_list if k in gene_set_to_use_dict}

out_path = model_gene_set_dict.keys()
# %%


tmp_ax = dose_data_tbl.query('DRUG_ID == @drug_id').assign(leading_edge = lambda df: df.SANGER_MODEL_ID.isin(credible_LE_cells_list)).groupby('leading_edge').AUC.plot.kde(legend=True)

plt.show()

# %%

agg_edge_df_list = []
agg_node_df_list = []

for tmp_gene_set_name in out_path:
    print(tmp_gene_set_name)
    tmp_gene_set = gene_set_to_use_dict[tmp_gene_set_name]
    tmp_tot_gene_set_leading_edge_list = tmp_res.query('Pathway_Name == @tmp_gene_set_name').Leading_Edge_Cell_Lines.iloc[0]
    tmp_gene_set_leading_edge_list = list(set(credible_LE_cells_list).intersection(tmp_tot_gene_set_leading_edge_list))
    node_df, edge_df = src.integration.leading_edge.construct_leading_edge_network(tmp_gene_set,tmp_gene_set_leading_edge_list,tmp_drug_excess_mutation_count_tbl)
    agg_edge_df_list.append(edge_df.assign(Pathway_Name = tmp_gene_set_name))
    agg_node_df_list.append(node_df.assign(Pathway_Name = tmp_gene_set_name))


# %%

obs_paths_list = pd.concat(agg_node_df_list).Pathway_Name.unique()
agg_node_df, agg_edge_df = src.integration.leading_edge.aggregate_pathway_networks_probabilistic_product(agg_node_df_list,agg_edge_df_list,obs_paths_list)

tmp_ax = agg_edge_df.assign(pr = lambda df: df.Consolidated_Edge_Weight.rank(pct=True)).plot(x='pr',y='Consolidated_Edge_Weight')
plt.show()


agg_G =  nx.from_pandas_edgelist(
        agg_edge_df.loc[:,['Source','Target','Consolidated_Edge_Weight']].rename(columns={'Consolidated_Edge_Weight':'weight'}), 
    source='Source', 
    target='Target', 
    edge_attr='weight' 
)

# %%
pos = src.integration.leading_edge.spectral_hilbert_layout(agg_G, gap_size=4)

# Plotting the result

exi,eyi,ezi = src.integration.leading_edge.generate_edge_contour_matrices(pos,agg_G,pd.DataFrame(pos).iloc[0,:].max(),resolution=500)

# %%
# interactive form for this visualisation using plotly

src.integration.leading_edge.create_interactive_network_explorer(pos,agg_node_df,exi,eyi,ezi,output_html_path=f'./img/ML_{drug_name}_c2_network.html')


# %%
node_size_dict = (agg_node_df.loc[:,['Gene','Consolidated_Intensity']].set_index('Gene').to_dict())['Consolidated_Intensity']
fig, ax = plt.subplots(figsize=(13, 12))  # Dark tech background
ax.set_facecolor('#090d16')
nx.draw_networkx_nodes(agg_G, pos, node_size=[10 ** ( node_size_dict[node]) for node in agg_G.nodes()], node_color='#4D96FF')
nx.draw_networkx_edges(agg_G, pos, alpha=0.5, edge_color='grey')
nx.draw_networkx_labels(agg_G, pos, font_size=10, font_color='black')
contour_filled = ax.contourf(exi, eyi, ezi, levels=20, cmap='plasma', alpha=0.8, zorder=1)
ax.set_xlim(-1, pd.DataFrame.from_dict(pos,orient='index',columns=['x','y']).x.max() + 1)
ax.set_ylim(-1, pd.DataFrame.from_dict(pos,orient='index',columns=['x','y']).y.max() + 1)
plt.title("Spectral (Fiedler Vector) Hilbert Layout")
plt.axis('off')
plt.show()




# %%
(
tmp_drug_excess_mutation_count_tbl
.query('~(sanger_model_id in @credible_LE_cells_list)')
.query('gene in @gene_in_optim_model_list')
.merge(
    pd.DataFrame(model_gene_set_dict.items()).explode(1).rename(columns={0:'Pathway',1:'gene'})
    .merge(best_ensemble.coef_df.iloc[0,:]
           .reset_index()
           .rename(columns={'index':'Pathway',0:'coef'}),how='left')
)
.assign(score = lambda df: df.coef * df.excess_mutation_count)
.groupby(['gene','sanger_model_id'])
.agg(weight = ('score','sum'))
.sort_values('weight',ascending=True)
.head(20)
)

# %%

