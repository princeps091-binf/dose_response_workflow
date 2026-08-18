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

from kneed import KneeLocator, find_shape
kneed_tbl = tmp_res.assign(x = lambda df: df.Min_mHG_P_Value.rank(pct=True),y=lambda df:-np.log10(df.Min_mHG_P_Value)).sort_values('x').loc[:,['Pathway_Name','x','y']]


direction, curve = find_shape(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy())
kl = KneeLocator(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy(), curve=curve, direction=direction)

out_path = kneed_tbl.query('x <= @kl.knee').Pathway_Name.to_list()

tmp_ax = kneed_tbl.plot(x='x',y='y')
tmp_ax.axvline(x=kl.knee, color="red", linestyle="--", linewidth=1.5, label="Knee Point")
plt.show()

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

leading_edge_member_list = tmp_res.loc[:,['Pathway_Name','Leading_Edge_Cell_Lines']].set_index('Pathway_Name').loc[gene_set_collection_excess_count_df.columns,'Leading_Edge_Cell_Lines'].to_list()

leading_edge_score_tbl = compute_leading_edge_quantiles_vectorized(gene_set_collection_excess_count_df,leading_edge_member_list)

# %%
tmp_path = "KEGG_ERBB_SIGNALING_PATHWAY"
tmp_LE_cells_list = tmp_res.query('Pathway_Name == @tmp_path ').Leading_Edge_Cell_Lines.iloc[0]
tmp_thresh = tmp_res.query('Pathway_Name == @tmp_path ').Optimal_Burden_Threshold_Tau.iloc[0]

tmp_path_AUC_mut_tbl= (gene_set_collection_excess_count_df.loc[:,tmp_path]
 .reset_index()
 .merge(
    dose_data_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID','AUC','sensitivity_p']],
    left_on='sanger_model_id',
    right_on='SANGER_MODEL_ID',
    how='left'
    )
 .dropna()
 .assign(LE = lambda df: np.where(df.sanger_model_id.isin(tmp_LE_cells_list),'red','grey'))
                       )
tmp_ax = (tmp_path_AUC_mut_tbl
 .plot
 .scatter(y='AUC',x=tmp_path,c='LE',s=50,alpha=0.1)
 # .groupby('LE')
 # .agg(m=(tmp_path,'mean'))
 )
plt.show()
 # %%
tmp_ax = (
        leading_edge_score_tbl.sum(axis=1).sort_values().reset_index().rename(columns= {0:'score'}).merge(
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

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

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


def compute_auprg(y_true, y_scores):
    """Calcule l'Aire Sous la Courbe Precision-Recall Gain (AUPRG) selon Flach & Kull (2015).
    y_true   : array-like, étiquettes réelles (0 ou 1)
    y_scores : array-like, probabilités ou scores prédits par le modèle
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    # 1. Calcul de la prévalence (pi)
    pi = np.mean(y_true)
    if pi == 0 or pi == 1:
        return 0.0  # Cas triviaux
    # 2. Obtenir la courbe PR standard de Scikit-Learn
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    # Inverser pour avoir recall croissant (de 0 à 1)
    precision = precision[::-1]
    recall = recall[::-1]
    # 3. Formules PR-Gain
    with np.errstate(divide="ignore", invalid="ignore"):
        rg = (recall - pi) / ((1 - pi) * recall)
        pg = (precision - pi) / ((1 - pi) * precision)
    # 4. Conserver uniquement les points où Recall Gain > 0 et Precision Gain > 0
    # (Ou effectuer l'interpolation linéaire vers l'origine (0,0))
    valid_mask = (recall > pi) & (precision > pi)
    rg_valid = rg[valid_mask]
    pg_valid = pg[valid_mask]
    if len(rg_valid) == 0:
        return 0.0  # Aucun gain par rapport au hasard
    # 5. Ancrer explicitement la courbe à l'origine (0, 0)
    rg_final = np.concatenate(([0.0], rg_valid))
    pg_final = np.concatenate(([0.0], pg_valid))
    # 6. S'assurer que les points sont strictement croissants sur Recall Gain
    sort_idx = np.argsort(rg_final)
    rg_sorted = rg_final[sort_idx]
    pg_sorted = pg_final[sort_idx]
    # 7. Intégration numérique par la méthode des trapèzes
    # Utiliser np.trapezoid (NumPy 2.0+) ou np.trapz (versions antérieures)
    auprg = np.trapezoid(pg_sorted, rg_sorted)
    return float(np.clip(auprg, 0.0, 1.0))

# %%
pr_auc_list = []
roc_auc_list = []
base_rate_list = []
auprg_list = []
thresh_span =np.concatenate((tmp_res.query('x < @kl.knee').x.to_numpy(), np.linspace(kl.knee,0.5,21)))
n_splits = 5
for tmp_thresh in thresh_span:
    out_path = tmp_res.query('x <= @tmp_thresh').Pathway_Name.to_list()
    tmp_drug_out_path_excess_count_df = gene_set_collection_excess_count_df.loc[:,out_path]
    out_path_leading_edge_score_tbl = leading_edge_score_tbl.loc[:,out_path]
    out_path_leading_edge_member_list = tmp_res.query('Pathway_Name in @out_path').Leading_Edge_Cell_Lines.explode().unique().tolist()
    cell_ids = tmp_drug_out_path_excess_count_df.index
    pathway_names = tmp_drug_out_path_excess_count_df.columns
    all_le_cells_set = set(c for sublist in out_path_leading_edge_member_list for c in sublist)
    LE_count_tbl = pd.DataFrame({'SANGER_MODEL_ID':out_path_leading_edge_member_list}).explode('SANGER_MODEL_ID').value_counts().reset_index().rename(columns={'count':'path_count'})
    LE_cells = LE_count_tbl.query('path_count > 0').SANGER_MODEL_ID.to_list()
    y_union = pd.Series(cell_ids.isin(LE_cells).astype(int), index=cell_ids)
    F_augmented_df = augment_features_for_or_logic(out_path_leading_edge_score_tbl,kneed_tbl)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Array to store Out-of-Fold (OOF) predicted probabilities
    base_rate_list.append(y_union.mean())
    oof_probs = np.zeros(len(y_union))
    for fold, (train_idx, val_idx) in enumerate(skf.split(F_augmented_df, y_union)):
        # Split data
        X_train, X_val = F_augmented_df.iloc[train_idx], F_augmented_df.iloc[val_idx]
        y_train, y_val = y_union.iloc[train_idx], y_union.iloc[val_idx]
        # Initialize ElasticNet Logistic Regression
        # clf = LogisticRegression(
        #     l1_ratio=0.5,  # Equal mix of L1 (Lasso) and L2 (Ridge)
        #     C=1.0,         # Inverse regularization strength
        #     solver="saga",
        #     max_iter=10000,
        #     random_state=42 + fold  # Vary random state per fold
        # )
        # Initialize Lasso Logistic Regression
        clf = LogisticRegression(
                    l1_ratio=1,          # Pure L1 (Lasso) regularization
                    solver="liblinear",    # Coordinate descent solver for small/medium sparse models
                    max_iter=1000,         # liblinear usually converges quickly (1000 is safe)
                    random_state=42 + fold  # Reproducibility per fold
                )
        # Fit model on training fold
        clf.fit(X_train, y_train)
        # Predict probabilities on validation fold
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
    # 3. Compute Out-of-Fold (OOF) Evaluation Metrics
    oof_preds = (oof_probs >= 0.5).astype(int)
    # Precision-Recall AUC (PR-AUC)
    precision, recall, _ = precision_recall_curve(y_union, oof_probs)
    oof_pr_auc = auc(recall, precision)
    # ROC-AUC
    oof_roc_auc = roc_auc_score(y_union, oof_probs)
    oof_auprg = compute_auprg(y_union,oof_preds)
    pr_auc_list.append(oof_pr_auc)
    roc_auc_list.append(oof_roc_auc)
    auprg_list.append(oof_auprg)
    print(f"{tmp_thresh:.2f} PR-AUC  : {oof_pr_auc:.4f}")
    print(f"{tmp_thresh:.2f} ROC-AUC : {oof_roc_auc:.4f}\n")

# %%

perf_summary_tbl = pd.DataFrame({'thresh':thresh_span,'ROC':roc_auc_list,'PR':pr_auc_list,'base_rate':base_rate_list,'auprg':auprg_list}).assign(norm_pr_auc = lambda df: (df.PR - df.base_rate)/(1-df.base_rate))

tmp_ax = perf_summary_tbl.plot('thresh','norm_pr_auc')
plt.show()

# %%
# Extract non-zero learned coefficients (Selected Pathways)
auprg_thresh = perf_summary_tbl.norm_pr_auc.max()
tmp_thresh = perf_summary_tbl.query('norm_pr_auc >= @auprg_thresh').sort_values('thresh',ascending=False).thresh.iloc[0]
out_path = tmp_res.query('x <= @tmp_thresh').Pathway_Name.to_list()

tmp_le = tmp_res.query('Pathway_Name in @out_path').Leading_Edge_Cell_Lines.explode().unique()

tmp_ax = dose_data_tbl.query('DRUG_ID == @drug_id').assign(LE = lambda df: np.where(df.SANGER_MODEL_ID.isin(tmp_le),'Leading_Edge','Rest')).groupby('LE').AUC.plot.kde(legend=True,title='AUC')
plt.show()

# %%
out_path_leading_edge_member_list = tmp_res.query('Pathway_Name in @out_path').Leading_Edge_Cell_Lines.to_list()

LE_count_tbl = pd.DataFrame({'SANGER_MODEL_ID':out_path_leading_edge_member_list}).explode('SANGER_MODEL_ID').value_counts().reset_index().rename(columns={'count':'path_count'})
# %%
import torch
print(torch.cuda.is_available())       # True if CUDA GPU is available
print(torch.cuda.get_device_name(0))   # GPU name

# %%
from tabpfn import TabPFNClassifier
import os
from dotenv import load_dotenv

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__name__))
env_path = os.path.join(script_dir, '.env')

# Load the specific path
load_dotenv(dotenv_path=env_path)
load_dotenv()
# %%
pr_auc_list = []
roc_auc_list = []
base_rate_list = []
n_splits = 5
for tmp_thresh in thresh_span:
    out_path = kneed_tbl.query('x <= @tmp_thresh').Pathway_Name.to_list()
    tmp_drug_out_path_excess_count_df = gene_set_collection_excess_count_df.loc[:,out_path]
    out_path_leading_edge_score_tbl = leading_edge_score_tbl.loc[:,out_path]
    out_path_leading_edge_member_list = tmp_res.query('Pathway_Name in @out_path').Leading_Edge_Cell_Lines.explode().unique().tolist()
    cell_ids = tmp_drug_out_path_excess_count_df.index
    pathway_names = tmp_drug_out_path_excess_count_df.columns
    all_le_cells_set = set(c for sublist in out_path_leading_edge_member_list for c in sublist)
    LE_count_tbl = pd.DataFrame({'SANGER_MODEL_ID':out_path_leading_edge_member_list}).explode('SANGER_MODEL_ID').value_counts().reset_index().rename(columns={'count':'path_count'})
    LE_cells = LE_count_tbl.query('path_count > 0').SANGER_MODEL_ID.to_list()
    y_union = pd.Series(cell_ids.isin(LE_cells).astype(int), index=cell_ids)
    base_rate_list.append(y_union.mean())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Array to store Out-of-Fold (OOF) predicted probabilities
    oof_probs = np.zeros(len(y_union))
    for fold, (train_idx, val_idx) in enumerate(skf.split(out_path_leading_edge_score_tbl, y_union)):
        # Split data
        X_train, X_val = out_path_leading_edge_score_tbl.iloc[train_idx], out_path_leading_edge_score_tbl.iloc[val_idx]
        y_train, y_val = y_union.iloc[train_idx], y_union.iloc[val_idx]
        # Initialize ElasticNet Logistic Regression
        model = TabPFNClassifier()
        model.fit(X_train, y_train)
        oof_probs[val_idx] = model.predict_proba(X_val)[:,1]
        # Fit model on training fold
    # 3. Compute Out-of-Fold (OOF) Evaluation Metrics
    # Precision-Recall AUC (PR-AUC)
    precision, recall, _ = precision_recall_curve(y_union, oof_probs)
    oof_pr_auc = auc(recall, precision)
    # ROC-AUC
    oof_roc_auc = roc_auc_score(y_union, oof_probs)
    print(f"{tmp_thresh:.2f} PR-AUC  : {oof_pr_auc:.4f}")
    print(f"{tmp_thresh:.2f} ROC-AUC : {oof_roc_auc:.4f}\n")
    pr_auc_list.append(oof_pr_auc)
    roc_auc_list.append(oof_roc_auc)

# %%

perf_summary_tbl = pd.DataFrame({'thresh':thresh_span,'ROC':roc_auc_list,'PR':pr_auc_list,'base_rate':base_rate_list}).assign(norm_pr_auc = lambda df: (df.PR - df.base_rate)/(1-df.base_rate))

tmp_ax = perf_summary_tbl.plot('thresh','norm_pr_auc')
plt.show()



# %%

from tabpfn_extensions.interpretability.shapiq import get_tabpfn_imputation_explainer
explainer = get_tabpfn_imputation_explainer(model=model, data=out_path_leading_edge_score_tbl)

sv = explainer.explain(out_path_leading_edge_score_tbl.iloc[2:3].values, budget=128)
print(sv)              # top interactions ranked by magnitude
sv.plot_waterfall()    # waterfall plot showing additive co

# %%
# Regression potential

tmp_ax =( dose_data_tbl
         .query('DRUG_ID == @drug_id')
         .merge(LE_count_tbl)
         .assign(LE = lambda df: pd.qcut(df.path_count,[0.01,0.1,0.25,0.5,0.75,0.95,1]))
         .groupby('LE').AUC.plot.kde(legend=True)
         )
plt.show()
# %%
tmp_ax= dose_data_tbl.query('DRUG_ID == @drug_id').merge(LE_count_tbl,how='left').fillna(0).assign(AUC_rank = lambda df: df.AUC.rank(pct=True,ascending=True)).plot.scatter(x='AUC_rank',y='path_count')
plt.show()
