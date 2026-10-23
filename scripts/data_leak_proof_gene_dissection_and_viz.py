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

gene_set_dict = src.utils.io.parse_gmt(gene_set_file)
Gene_Set_size_tbl = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).explode('Genes').Gene_Set.value_counts().reset_index().rename(columns={'count':'gene_count'})

sub_collection_list = ['REACTOME','KEGG','PID','WP']
collection_to_use_list = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).assign(collection = lambda df: [i.split('_')[0] for i in df.Gene_Set]).query('collection in @sub_collection_list').Gene_Set.drop_duplicates().to_list()

gene_set_to_use_dict = {k: gene_set_dict[k] for k in collection_to_use_list if k in gene_set_dict}

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


class OutOfFoldEnsemble:
    """Wraps K fold models to predict averaged out-of-fold probabilities on new data."""
    def __init__(self, models, optuna_params, coef_df, LE_cells_list):
        self.models = models
        self.optuna = optuna_params
        self.coef_df = coef_df
        self.LE_cell = LE_cells_list
    
    def _resolve_fold_inputs(self,X):
        return list(zip(self.models, X))
    
    def predict_proba(self, X):
        model_input_pairs = self._resolve_fold_inputs(X)
        # Gather predicted positive-class probabilities from each fold model with its corresponding data
        fold_probs = np.column_stack([
            model.predict_proba(X_fold)[:, 1] 
            for model, X_fold in model_input_pairs
        ])
        # Average across all K models
        mean_probs = np.mean(fold_probs, axis=1)
        # Return standard 2D probability array [P(y=0), P(y=1)]
        return np.column_stack([1 - mean_probs, mean_probs])
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)





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

drug_id = 1012
tmp_drug_gene_excess_mutation_count_tbl = src.mutation.gene_set_analysis.get_excess_mutation_count_matrix(drug_id,K_matrix,N_vector,dose_data_tbl,all_cells,all_genes)

tmp_drug_data_tbl = dose_data_tbl.query('DRUG_ID == @drug_id')
drug_name = tmp_drug_data_tbl.DRUG_NAME.iloc[0]

gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_gene_excess_mutation_count_tbl,all_cells,all_genes,gene_set_to_use_dict)

shared_cell_id_list = list(set(tmp_drug_data_tbl.SANGER_MODEL_ID.unique().tolist()).intersection(gene_set_collection_excess_count_df.index.tolist())) 

tmp_drug_sensitivity_series = tmp_drug_data_tbl.query('SANGER_MODEL_ID in @shared_cell_id_list').set_index('SANGER_MODEL_ID').loc[shared_cell_id_list,'sensitivity_p']

tmp_drug_excess_mutation_count_tbl = gene_set_collection_excess_count_df.loc[shared_cell_id_list,:]

# %%

import joblib
best_ensemble_path = f'./data/tmp_res/leak_free_logit_ensemble_drug_{drug_id}.joblib'

best_ensemble = joblib.load(best_ensemble_path)

# %%


pos_features =  (best_ensemble.coef_df
                 .query('coef > 0')
                 .Features.value_counts()
                 .reset_index()
                 # .query('count > 1')
                 .Features.to_list()
                 )
pos_features_gene_content_list = [gene_set_to_use_dict[i] for i in pos_features if i in gene_set_to_use_dict.keys()]

unique_pos_genes_elements = list(set(item for sublist in pos_features_gene_content_list for item in sublist))
# %%

n_splits = len(best_ensemble.models)


best_thresh = best_ensemble.optuna['tmp_thresh']
best_C = best_ensemble.optuna['C']
fold_leading_edge_score_tbl_list = []
best_in_data_tbl = []
for tmp_fold in range(n_splits):
    # --- FEATURE SELECTION & TARGET CONSTRUCTION ---
    # --- Need to produce the minimum hypergeometric score within the fold
    tmp_fold_features = best_ensemble.models[tmp_fold].feature_names_in_
    tmp_fold_gene_set_features_dict = {k: gene_set_to_use_dict[k] for k in tmp_fold_features if k in gene_set_to_use_dict.keys()}
    gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_gene_excess_mutation_count_tbl,all_cells,all_genes,tmp_fold_gene_set_features_dict)
    fold_drug_excess_mutation_count_tbl = gene_set_collection_excess_count_df.loc[shared_cell_id_list,:]
    fold_leading_edge_score_tbl = compute_leading_edge_quantiles_vectorized(fold_drug_excess_mutation_count_tbl,best_ensemble.LE_cell[tmp_fold].set_index('Pathway_Name').loc[fold_drug_excess_mutation_count_tbl.columns,:].Leading_Edge_Cell_Lines.tolist())
    fold_leading_edge_score_tbl_list.append(fold_leading_edge_score_tbl)
    # best_in_data_tbl.append(augment_features_for_or_logic(fold_leading_edge_score_tbl, best_ensemble.LE_cell[tmp_fold]))
    best_in_data_tbl.append(fold_leading_edge_score_tbl)

# %%
new_predictions = best_ensemble.predict_proba(best_in_data_tbl)[:, 1]

new_pred_df = pd.DataFrame({'SANGER_MODEL_ID':best_in_data_tbl[0].index,'proba':new_predictions})

tmp_drug_sensitivity_series.reset_index().merge(new_pred_df).assign(sensitive = lambda df: df.sensitivity_p.lt(0.5), p_class = lambda df: df.proba.gt(0.5)).groupby('p_class').sensitivity_p.plot.kde(legend=True)
plt.show()


# %%


def predict_proba_by_fold(best_ensemble_obj,X):
    model_input_pairs = best_ensemble_obj._resolve_fold_inputs(X)
    # Gather predicted positive-class probabilities from each fold model with its corresponding data
    fold_probs = np.column_stack([
        model.predict_proba(X_fold)[:, 1] 
        for model, X_fold in model_input_pairs
    ])
    return fold_probs,best_in_data_tbl[0].index

fold_proba_list = predict_proba_by_fold(best_ensemble,best_in_data_tbl)

# %%
fold_proba_df = []
for idx in range(fold_proba_list[0].shape[1]):
    fold_proba_df.append(pd.DataFrame({'sanger_model_id':fold_proba_list[1],'proba':fold_proba_list[0][:,idx]}).assign(fold = idx))

fold_proba_df = pd.concat(fold_proba_df)

# %%
from collections import defaultdict
gene_to_path_dict = defaultdict(list)

# Loop through and populate the new dictionary
for key, value_list in gene_set_to_use_dict.items():
    for value in value_list:
        gene_to_path_dict[value].append(key)


# Helper function for fast lookup against the FIXED LE reference scale
def fixed_ecdf_rank(vals: np.ndarray,le_ref_values) -> np.ndarray:
    return np.searchsorted(le_ref_values, vals, side='right') / n_le

# %%
res_df = []
edge_raw_df = []
for tmp_gene in unique_pos_genes_elements:
    for tmp_fold in range(n_splits):
        tmp_fold_nz_features = best_ensemble.coef_df.query('fold == @tmp_fold and coef != 0').Features.tolist()
        tmp_gene_features = gene_to_path_dict[tmp_gene]
        tmp_fold_proba_tbl = pd.DataFrame({'sanger_model_id':fold_proba_list[1],'proba':fold_proba_list[0][:,tmp_fold]})
        tmp_features_to_keep_list = list(set(tmp_gene_features).intersection(tmp_fold_nz_features))
        print(len(tmp_features_to_keep_list))
        if len(tmp_features_to_keep_list) > 0 :
            tmp_fold_gene_set_features_dict = {k: gene_set_to_use_dict[k] for k in tmp_features_to_keep_list}
            ablated_gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_gene_excess_mutation_count_tbl.query('gene != @tmp_gene'),all_cells,all_genes,tmp_fold_gene_set_features_dict)
            
            ablated_tmp_drug_excess_mutation_count_tbl = ablated_gene_set_collection_excess_count_df.loc[shared_cell_id_list,:]
            
            original_drug_excess_mutation_count_tbl = tmp_drug_excess_mutation_count_tbl.loc[shared_cell_id_list,tmp_features_to_keep_list]
            delta_Q_df = pd.DataFrame(
                    0.0, 
                    index=ablated_tmp_drug_excess_mutation_count_tbl.index, 
                    columns=tmp_features_to_keep_list
                )
            for pathway in tmp_features_to_keep_list:
                le_cells = best_ensemble.LE_cell[tmp_fold].set_index('Pathway_Name').loc[pathway,:].Leading_Edge_Cell_Lines
                if not le_cells:
                    continue
                    
                # 1. Extract and sort the FIXED LE Reference values for this pathway from df_orig
                le_ref_values = np.sort(original_drug_excess_mutation_count_tbl.loc[le_cells, pathway].values)
                n_le = len(le_ref_values)
                
                if n_le == 0:
                    continue
                # 2. Get baseline quantile rank (Q_orig) on fixed LE reference
                orig_scores = original_drug_excess_mutation_count_tbl[pathway].values
                Q_orig = fixed_ecdf_rank(orig_scores,le_ref_values)
                # 3. Get ablated quantile rank (Q_ablated) against the EXACT SAME fixed LE reference
                ablated_scores = ablated_tmp_drug_excess_mutation_count_tbl[pathway].values
                Q_ablated = fixed_ecdf_rank(ablated_scores,le_ref_values)
                # 4. Net Marginal Quantile Shift
                delta_Q_df[pathway] = Q_orig - Q_ablated   
            delta_Q_df = delta_Q_df.reset_index().merge(tmp_fold_proba_tbl.reset_index(drop=True).rename(columns={'SANGER_MODEL_ID':'sanger_model_id'})).set_index('sanger_model_id')
            delta_Q_sum_df = delta_Q_df.drop('proba',axis=1).mul(delta_Q_df['proba'], axis=0).sum(axis=0).div(delta_Q_df.proba.sum()).reset_index().assign(gene= tmp_gene,fold=tmp_fold).rename(columns={'index':'Features',0:'weighted_QE_score'}).merge(best_ensemble.coef_df.query('Features in @tmp_features_to_keep_list')).assign(tot = lambda df: df.weighted_QE_score * df.coef)
            res_df.append(delta_Q_sum_df)
            edge_raw_df.append(delta_Q_df.drop('proba',axis=1).reset_index().melt(id_vars='sanger_model_id').query('value > 0').assign(gene = tmp_gene,fold=tmp_fold))

# %%

pd.concat(res_df).groupby(['gene','Features']).agg(tot=('tot','sum')).reset_index().groupby('gene').agg(mtot = ('tot','max')).sort_values('mtot').query('mtot > 0 ')

# %%
candidate_cell_gene_combo_df = pd.concat(edge_raw_df).rename(columns={'variable':'Features'}).merge(best_ensemble.coef_df).assign(score = lambda df: df.value * df.coef).merge(fold_proba_df).query('proba > 0.5').groupby(['sanger_model_id','Features','gene']).agg(mscore = ('score','sum')).reset_index().groupby(['sanger_model_id','gene']).agg(mscore = ('mscore','max')).reset_index().query('mscore > 0 ')

# %%

wide_contrib = candidate_cell_gene_combo_df.pivot(
        index='sanger_model_id', 
        columns='gene', 
        values='mscore'
    ).fillna(0.0)

# %% 
# 5. Compute shared counts via matrix multiplication: (G_genes x N_cells) @ (N_cells x G_genes)
matrix_vals = wide_contrib.values
co_occurrence_matrix = np.dot(matrix_vals.T, matrix_vals)

# 6. Format back into a tidy edge list DataFrame
genes = wide_contrib.columns.tolist()
adj_df = pd.DataFrame(co_occurrence_matrix, index=genes, columns=genes)

edges = (
    adj_df.stack()
    .reset_index(name='shared_sensitive_cells')
    .rename(columns={'level_0': 'gene_A', 'level_1': 'gene_B'})
).query('shared_sensitive_cells > 0 and gene_A != gene_B')

# %%
filtering_edge_df = edges.sort_values('shared_sensitive_cells',ascending=False).reset_index(drop=True).assign(cum_sum = lambda df: df.shared_sensitive_cells.cumsum())

total_weight = filtering_edge_df['shared_sensitive_cells'].sum()
threshold_mass = total_weight * 0.75
cutoff_idx = filtering_edge_df.cum_sum.gt(threshold_mass).idxmax()
filtered_edges_df = filtering_edge_df.iloc[:cutoff_idx + 1].copy()
filtered_edges_df = filtered_edges_df.assign(score_rank = lambda df: df.shared_sensitive_cells.rank(pct=True))
# %%
#
# tmp_ax = edges.assign(pr = lambda df: df.shared_sensitive_cells.rank(pct=True)).sort_values('shared_sensitive_cells').plot(x='pr',y='shared_sensitive_cells')
# plt.show()
#
# # %%
# from kneed import KneeLocator
# # 1. Sort the edge weights in descending order
# sorted_weights = np.sort(edges['shared_sensitive_cells'].values)[::-1]
# x_vals = np.arange(len(sorted_weights))
# from kneed import KneeLocator
# kneedle = KneeLocator(
#     x_vals, 
#     sorted_weights, 
#     curve='convex', 
#     direction='decreasing',
#     interp_method='polynomial'
# )
#
# # 3. Extract the threshold weight at the knee point
# threshold_weight = sorted_weights[kneedle.knee]
#
# # 4. Subset your original DataFrame to keep only the salient edges
# df_salient = edges[edges['shared_sensitive_cells'] >= threshold_weight].copy()
#
# %%

agg_G =  nx.from_pandas_edgelist(
        filtered_edges_df.loc[:,['gene_A','gene_B','shared_sensitive_cells']].rename(columns={'shared_sensitive_cells':'weight'}), 
    source='gene_A', 
    target='gene_B', 
    edge_attr='weight' 
)

# %%

agg_node_df = pd.concat(res_df).groupby(['gene','Features']).agg(tot=('tot','sum')).reset_index().groupby(['gene','Features']).agg(mtot = ('tot','sum')).reset_index().groupby('gene').agg(weight = ('mtot','max')).reset_index().rename(columns={'gene':'Gene'}).assign(Consolidated_Intensity = lambda df: df.weight.rank(pct=True))
# %%

# Build graph just for layout based on pathway affiliation
shared_pathway_counts = defaultdict(int)
for pathway in best_ensemble.coef_df.query('coef != 0').Features.to_list():
    if pathway in gene_set_to_use_dict.keys():
        # Filter constitutive genes to keep only valid positive contributors
        genes_in_pathway = [g for g in gene_set_to_use_dict[pathway] if g in agg_G.nodes()]
        # Generate all unique pairs within this pathway
        n_genes = len(genes_in_pathway)
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                # Sort alphabetically to ensure undirected uniqueness (A-B == B-A)
                pair = tuple(sorted((genes_in_pathway[i], genes_in_pathway[j])))
                shared_pathway_counts[pair] += 1

edge_data = []
for (g1, g2), count in shared_pathway_counts.items():
    if count >= 1:
        edge_data.append({
            'gene_A': g1,
            'gene_B': g2,
            'shared_pathways': count
        })

pos_edges_df = pd.DataFrame(edge_data)
if not pos_edges_df.empty:
    pos_edges_df = pos_edges_df.sort_values(by='shared_pathways', ascending=False).reset_index(drop=True)

# %%
pos_G =  nx.from_pandas_edgelist(
        pos_edges_df.loc[:,['gene_A','gene_B','shared_pathways']].rename(columns={'shared_pathways':'weight'}), 
    source='gene_A', 
    target='gene_B', 
    edge_attr='weight' 
)

pos = src.integration.leading_edge.spectral_hilbert_layout(pos_G, gap_size=10)

# %%

exi,eyi,ezi = src.integration.leading_edge.generate_edge_contour_matrices(pos,agg_G,pd.DataFrame(pos).iloc[0,:].max(),resolution=100)

# %%
# interactive form for this visualisation using plotly

src.integration.leading_edge.create_interactive_network_explorer(pos,agg_node_df,exi,eyi,ezi,output_html_path=f'./img/data_leak_proof_{drug_name}_c2_network.html')

