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
leading_edge_member_list = tmp_res.loc[:,['Pathway_Name','Leading_Edge_Cell_Lines']].set_index('Pathway_Name').loc[gene_set_collection_excess_count_df.columns,'Leading_Edge_Cell_Lines'].to_list()

leading_edge_score_tbl = compute_leading_edge_quantiles_vectorized(gene_set_collection_excess_count_df,leading_edge_member_list)

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

import optuna
# 1. Define bounds for tmp_thresh based on your original thresh_span
thresh_min = float(tmp_res.x.min())
thresh_max = float(0.5)

def logit_objective(trial):
    # --- HYPERPARAMETER SAMPLING ---
    # Sample tmp_thresh continuously between the bounds of thresh_span
    tmp_thresh = trial.suggest_float("tmp_thresh", thresh_min, thresh_max)
    # Sample l1_ratio between 0 (Pure L2) and 1 (Pure L1 / Lasso)
    # l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    # Optionally tune C (inverse regularization strength) alongside ElasticNet
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
# --- FEATURE SELECTION & TARGET CONSTRUCTION ---
    out_path = tmp_res.query('x <= @tmp_thresh').Pathway_Name.to_list()
# Prune search early if threshold selects zero pathways
    if len(out_path) == 0:
        return 0.0  # Return baseline low score
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
        pd.DataFrame({'SANGER_MODEL_ID': out_path_leading_edge_member_list})
        .explode('SANGER_MODEL_ID')
        .value_counts()
        .reset_index()
        .rename(columns={'count': 'path_count'})
    )
    LE_cells = LE_count_tbl.query('path_count > 0').SANGER_MODEL_ID.to_list()
    y_union = pd.Series(cell_ids.isin(LE_cells).astype(int), index=cell_ids)
# Check for single-class targets in extreme threshold edge cases
    if y_union.nunique() < 2:
        return 0.0
    F_augmented_df = augment_features_for_or_logic(out_path_leading_edge_score_tbl, tmp_res)
# --- OUT-OF-FOLD (OOF) CROSS-VALIDATION ---
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_union))
    for fold, (train_idx, val_idx) in enumerate(skf.split(F_augmented_df, y_union)):
        X_train, X_val = F_augmented_df.iloc[train_idx], F_augmented_df.iloc[val_idx]
        y_train, y_val = y_union.iloc[train_idx], y_union.iloc[val_idx]
        clf = LogisticRegression(
            l1_ratio=1,  # Equal mix of L1 (Lasso) and L2 (Ridge)
            C=C,         # Inverse regularization strength
            solver="liblinear",
            max_iter=10000,
            random_state=42 + fold  # Vary random state per fold
        )
        clf.fit(X_train, y_train)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
# --- OOF EVALUATION METRICS ---
    # Compute AUPRG on continuous predicted probabilities (not binary oof_preds)
    oof_auprg = compute_auprg(y_union, oof_probs)
    # Store auxiliary metrics as trial user attributes for later retrieval
    precision, recall, _ = precision_recall_curve(y_union, oof_probs)
    trial.set_user_attr("oof_pr_auc", auc(recall, precision))
    trial.set_user_attr("oof_roc_auc", roc_auc_score(y_union, oof_probs))
    trial.set_user_attr("base_rate", float(y_union.mean()))
    return oof_auprg

# %%
# --- EXECUTE OPTUNA STUDY ---
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=123)
)

study.optimize(logit_objective, n_trials=50, timeout=1800)  # Adjust trials/timeout as needed

# %%

print(f"Best OOF AUPRG   : {study.best_value:.4f}")
print(f"Best tmp_thresh  : {study.best_params['tmp_thresh']:.4f}")
print(f"Associated PR-AUC: {study.best_trial.user_attrs['oof_pr_auc']:.4f}")
print(f"Associated ROC-AUC: {study.best_trial.user_attrs['oof_roc_auc']:.4f}")

# %%
import plotly.io as pio
pio.renderers.default = "browser"
fig = optuna.visualization.plot_contour(study)
fig.show()

# %%

perf_df = pd.DataFrame([f.user_attrs for f in study.trials]).assign(thresh = [f.params['tmp_thresh'] for f in study.trials], auprg = [f.values[0] for f in study.trials])

tmp_ax = perf_df.sort_values('thresh').plot('thresh','auprg')
plt.show()

# %%
# Recover best model


# 1. Custom class to wrap the K fold models into a unified ensemble
class OutOfFoldEnsemble:
    """Wraps K fold models to predict averaged out-of-fold probabilities on new data."""
    def __init__(self, models):
        self.models = models
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
# 2. Extract best parameters from your Optuna study
best_params = study.best_params
best_tmp_thresh = best_params["tmp_thresh"]
best_C = best_params["C"]

# 3. Reconstruct feature matrix and target using best_tmp_thresh
out_path = tmp_res.query('x <= @best_tmp_thresh').Pathway_Name.to_list()
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

# 4. Fit and collect the K fold models
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_fold_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(F_augmented_df, y_union)):
    X_train, y_train = F_augmented_df.iloc[train_idx], y_union.iloc[train_idx]
    # Initialize model using best hyperparameter values
    clf = LogisticRegression(
            l1_ratio=1,  # Equal mix of L1 (Lasso) and L2 (Ridge)
            C=best_C,         # Inverse regularization strength
            solver="liblinear",
            max_iter=10000,
            random_state=42 + fold  # Vary random state per fold
    )
    clf.fit(X_train, y_train)
    best_fold_models.append(clf)

# 5. Instantiate the ensemble
import joblib
best_ensemble = OutOfFoldEnsemble(best_fold_models)
joblib.dump(best_ensemble, f"./data/tmp_res/logit_ensemble_drug_{drug_id}.joblib", compress=3)  # compress level 0-9
# %%
# Usage example on new unseen samples / holdout data:
new_predictions = best_ensemble.predict_proba(F_augmented_df)[:, 1]

optim_pred_df = pd.DataFrame({'proba':new_predictions,'sanger_model_id':F_augmented_df.index.tolist()}).merge(y_union.reset_index()).rename(columns={0:'LE'})

tmp_ax = optim_pred_df.plot.scatter(x='proba',y='LE',alpha=0.1)
plt.show()

# %%


optim_pred_df = optim_pred_df.merge(
        dose_data_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID','AUC','sensitivity_p']].rename(columns={'SANGER_MODEL_ID':'sanger_model_id'})
        )

tmp_ax = optim_pred_df.plot.scatter(x='proba',y='AUC',alpha=0.4)
plt.show()
# %%
# Extract non-zero coefficients across all fold models
coef_df = pd.DataFrame(
    [model.coef_[0] for model in best_ensemble.models],
    columns=F_augmented_df.columns
)

# Mean weight and selection frequency across folds
summary_coefs = pd.DataFrame({
    'mean_coef': coef_df.mean(axis=0),
    'min_coef': coef_df.min(axis=0),
    'max_coef': coef_df.max(axis=0),
    'selection_freq': (coef_df != 0).mean(axis=0)
}).sort_values(by='selection_freq', ascending=False)

summary_coefs.query('selection_freq >0').sort_values('mean_coef',ascending=True).loc[:,['min_coef','max_coef','mean_coef']]

# %%

tmp_ax = dose_data_tbl.query('DRUG_ID == @drug_id').assign(LE = lambda df: np.where(df.SANGER_MODEL_ID.isin(LE_cells),'Leading_Edge','Rest')).groupby('LE').AUC.plot.kde(legend=True,title='AUC')
plt.show()

# %%




# %%
def tabpfn_objective(trial):
    # --- HYPERPARAMETER SAMPLING ---
    # Sample tmp_thresh continuously between the bounds of thresh_span
    tmp_thresh = trial.suggest_float("tmp_thresh", thresh_min, thresh_max)
# --- FEATURE SELECTION & TARGET CONSTRUCTION ---
    out_path = tmp_res.query('x <= @tmp_thresh').Pathway_Name.to_list()
# Prune search early if threshold selects zero pathways
    if len(out_path) == 0:
        return 0.0  # Return baseline low score
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
        pd.DataFrame({'SANGER_MODEL_ID': out_path_leading_edge_member_list})
        .explode('SANGER_MODEL_ID')
        .value_counts()
        .reset_index()
        .rename(columns={'count': 'path_count'})
    )
    LE_cells = LE_count_tbl.query('path_count > 0').SANGER_MODEL_ID.to_list()
    y_union = pd.Series(cell_ids.isin(LE_cells).astype(int), index=cell_ids)
# Check for single-class targets in extreme threshold edge cases
    if y_union.nunique() < 2:
        return 0.0
    F_augmented_df = augment_features_for_or_logic(out_path_leading_edge_score_tbl, tmp_res)
# --- OUT-OF-FOLD (OOF) CROSS-VALIDATION ---
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_union))
    for fold, (train_idx, val_idx) in enumerate(skf.split(F_augmented_df, y_union)):
        X_train, X_val = F_augmented_df.iloc[train_idx], F_augmented_df.iloc[val_idx]
        y_train, y_val = y_union.iloc[train_idx], y_union.iloc[val_idx]
        model = TabPFNClassifier()
        model.fit(X_train, y_train)
        oof_probs[val_idx] = model.predict_proba(X_val)[:,1]
# --- OOF EVALUATION METRICS ---
    # Compute AUPRG on continuous predicted probabilities (not binary oof_preds)
    oof_auprg = compute_auprg(y_union, oof_probs)
    # Store auxiliary metrics as trial user attributes for later retrieval
    precision, recall, _ = precision_recall_curve(y_union, oof_probs)
    trial.set_user_attr("oof_pr_auc", auc(recall, precision))
    trial.set_user_attr("oof_roc_auc", roc_auc_score(y_union, oof_probs))
    trial.set_user_attr("base_rate", float(y_union.mean()))
    return oof_auprg

# %%
# --- EXECUTE OPTUNA STUDY ---
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=123)
)

study.optimize(tabpfn_objective, n_trials=50, timeout=1800)  # Adjust trials/timeout as needed

# %%

print(f"Best OOF AUPRG   : {study.best_value:.4f}")
print(f"Best tmp_thresh  : {study.best_params['tmp_thresh']:.4f}")
print(f"Associated PR-AUC: {study.best_trial.user_attrs['oof_pr_auc']:.4f}")
print(f"Associated ROC-AUC: {study.best_trial.user_attrs['oof_roc_auc']:.4f}")

# %%
import plotly.io as pio
pio.renderers.default = "browser"
fig = optuna.visualization.plot_contour(study)
fig.show()

# %%

perf_df = pd.DataFrame([f.user_attrs for f in study.trials]).assign(thresh = [f.params['tmp_thresh'] for f in study.trials], auprg = [f.values[0] for f in study.trials])

tmp_ax = perf_df.sort_values('thresh').plot('thresh','auprg')
plt.show()

# %%
# 2. Extract best parameters from your Optuna study
best_params = study.best_params
best_tmp_thresh = best_params["tmp_thresh"]

# 3. Reconstruct feature matrix and target using best_tmp_thresh
out_path = tmp_res.query('x <= @best_tmp_thresh').Pathway_Name.to_list()
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
# 4. Fit and collect the K fold models
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_fold_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(F_augmented_df, y_union)):
    X_train, X_val = F_augmented_df.iloc[train_idx], F_augmented_df.iloc[val_idx]
    y_train, y_val = y_union.iloc[train_idx], y_union.iloc[val_idx]
    model = TabPFNClassifier(fit_mode = 'fit_with_cache')
    model.fit(X_train, y_train)
    best_fold_models.append(model)

# 5. Instantiate the ensemble
best_ensemble = OutOfFoldEnsemble(best_fold_models)

# %%
# Usage example on new unseen samples / holdout data:
new_predictions = best_ensemble.predict_proba(F_augmented_df)[:, 1]

optim_pred_df = pd.DataFrame({'proba':new_predictions,'sanger_model_id':F_augmented_df.index.tolist()}).merge(y_union.reset_index()).rename(columns={0:'LE'})

tmp_ax = optim_pred_df.plot.scatter(x='proba',y='LE',alpha=0.1)
plt.show()

# %%
optim_pred_df = optim_pred_df.merge(
        dose_data_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID','AUC','sensitivity_p']].rename(columns={'SANGER_MODEL_ID':'sanger_model_id'})
        )

tmp_ax = optim_pred_df.plot.scatter(x='proba',y='AUC',alpha=0.1)
plt.show()
# %%
from tabpfn_extensions.interpretability.shapiq import get_tabpfn_imputation_explainer
model = best_fold_models[0]
explainer = get_tabpfn_imputation_explainer(model=model, data=F_augmented_df)

sv = explainer.explain(F_augmented_df.iloc[2:3].values, budget=64)
print(sv)              # top interactions ranked by magnitude
sv.plot_waterfall()    # waterfall plot showing additive co

# %%

# %%
# Regression potential

tmp_ax =( dose_data_tbl
         .query('DRUG_ID == @drug_id')
         .merge(LE_count_tbl.rename(columns={'Leading_Edge_Cell_Lines':'SANGER_MODEL_ID'}))
         .assign(LE = lambda df: pd.qcut(df.path_count,[0.01,0.1,0.25,0.5,0.75,0.95,1]))
         .groupby('LE').AUC.plot.kde(legend=True)
         )
plt.show()
# %%
tmp_ax= dose_data_tbl.query('DRUG_ID == @drug_id').merge(LE_count_tbl.rename(columns={'Leading_Edge_Cell_Lines':'SANGER_MODEL_ID'}),how='left').fillna(0).assign(AUC_rank = lambda df: df.AUC.rank(pct=True,ascending=True)).plot.scatter(x='AUC_rank',y='path_count')
plt.show()
