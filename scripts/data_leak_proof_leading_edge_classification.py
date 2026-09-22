
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

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


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

def compute_fold_mhg_and_features(
    raw_excess_df, 
    drug_sensitivity_df, 
    tmp_thresh, 
    train_idx, 
    val_idx
):
    """
    Computes xlmhgt, mHG pathway pruning, target construction (y_union), 
    and leading-edge percentile feature ranks STRICTLY on the training fold.
    """
    # ensure correspondence of cell line ordering between the dose response and the excess mutation tables
    # -> convert the idx into actual cell line IDs?
    # 1. Split Raw Inputs into Train and Validation
    raw_train = raw_excess_df.iloc[train_idx]
    y_sens_train = drug_sensitivity_df.sensitivity_p.iloc[train_idx]
    
    raw_val = raw_excess_df.iloc[val_idx]
    
    # 2. Run xlmhgt STRICTLY on Training Fold
    # Sort training samples by drug sensitivity (descending)
    train_rank_order = y_sens_train.sort_values(ascending=False).index
    
    
    fold_res_df = src.mutation.gene_set_analysis.run_high_throughput_parallel_xlmhg(
        pathway_burden_df = raw_train,   
        drug_sensitivity_df = y_sens_train, 
        n_burden_steps = 20,
        auc_col = 'sensitivity_p',
        sanger_id_col = 'SANGER_MODEL_ID',
        n_jobs = 8  
    )

    tmp_res = tmp_res.assign(x = lambda df: df.Min_mHG_P_Value.rank(pct=True),y=lambda df:-np.log10(df.Min_mHG_P_Value)).sort_values('x')

    # 3. Filter Pathways based on Trial Threshold (tmp_thresh)
    out_path = fold_res_df.query('x <= @tmp_thresh').Pathway_Name.tolist()
    if len(out_path) == 0:
        return None, None, None, None
        
    # 4. Construct Target (y_union) for Training and Validation Folds
    train_le_cells = (
        fold_res_df.query('Pathway_Name in @out_path')
        .Leading_Edge_Cell_Lines.explode()
        .unique()
    )
    
    y_train = pd.Series(raw_train.index.isin(train_le_cells).astype(int), index=raw_train.index)
    y_val = pd.Series(raw_val.index.isin(train_le_cells).astype(int), index=raw_val.index)
    
    # Check for single-class target in training fold
    if y_train.nunique() < 2:
        return None, None, None, None

    # 5. Build Leading-Edge Percentile Rank Features (eCDF Fit on Train LE Only)
    X_train_list, X_val_list = [], []
    
    for p in out_path:
        p_le_cells = fold_res_df.query('Pathway_Name == @p').Leading_Edge_Cell_Lines.values[0]
        
        # Fit eCDF strictly on training fold's leading-edge excess scores
        train_le_scores = raw_train.loc[raw_train.index.isin(p_le_cells), p].values
        if len(train_le_scores) == 0:
            train_le_scores = raw_train[p].values  # Fallback if empty
            
        sorted_le_scores = np.sort(train_le_scores)
        
        # Transform Train features
        train_p_scores = raw_train[p].values
        train_perc = np.searchsorted(sorted_le_scores, train_p_scores, side="right") / len(sorted_le_scores)
        X_train_list.append(pd.Series(train_perc, index=raw_train.index, name=p))
        
        # Transform Validation features using Training-learned eCDF
        val_p_scores = raw_val[p].values
        val_perc = np.searchsorted(sorted_le_scores, val_p_scores, side="right") / len(sorted_le_scores)
        X_val_list.append(pd.Series(val_perc, index=raw_val.index, name=p))
        
    X_train = pd.concat(X_train_list, axis=1)
    X_val = pd.concat(X_val_list, axis=1)
    
    # Optional feature augmentation step (if required by logic)
    X_train = augment_features_for_or_logic(X_train, fold_res_df)
    X_val = augment_features_for_or_logic(X_val, fold_res_df)
    
    return X_train, y_train, X_val, y_val


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
    # Optionally tune C (inverse regularization strength) alongside ElasticNet
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)

# Outer Cross-Validation on RAW Un-transformed Data
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(gene_set_collection_excess_count_df))
    oof_targets = np.zeros(len(gene_set_collection_excess_count_df))
    valid_fold_indices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(gene_set_collection_excess_count_df, tmp_drug_data_tbl)):
        X_train_df = tmp_drug_excess_mutation_count_tbl.iloc[train_idx]
        Y_train_df = tmp_drug_sensitivity_series.iloc[train_idx]
        X_val = tmp_drug_excess_mutation_count_tbl.iloc[val_idx]
# --- FEATURE SELECTION & TARGET CONSTRUCTION ---
# --- Need to produce the minimum hypergeometric score within the fold
        tmp_res = src.mutation.gene_set_analysis.run_high_throughput_parallel_xlmhg(
            pathway_burden_df = X_train_df,   
            drug_sensitivity_df = Y_train_df.reset_index(), 
            n_burden_steps = 20,
            auc_col = 'sensitivity_p',
            sanger_id_col = 'SANGER_MODEL_ID',
            n_jobs = 8  
        )

        tmp_res = tmp_res.assign(x = lambda df: df.Min_mHG_P_Value.rank(pct=True),y=lambda df:-np.log10(df.Min_mHG_P_Value)).sort_values('x')

        out_path = tmp_res.query('x <= @tmp_thresh').Pathway_Name.to_list()
        leading_edge_member_list = tmp_res.loc[:,['Pathway_Name','Leading_Edge_Cell_Lines']].set_index('Pathway_Name').loc[X_train_df.columns,'Leading_Edge_Cell_Lines'].to_list()
        leading_edge_score_tbl = compute_leading_edge_quantiles_vectorized(X_train_df,leading_edge_member_list)
        # Prune search early if threshold selects zero pathways
        if len(out_path) == 0:
            return 0.0  # Return baseline low score
        out_path_leading_edge_score_tbl = leading_edge_score_tbl.loc[:, out_path]
        out_path_leading_edge_member_list = (
            tmp_res.query('Pathway_Name in @out_path')
            .Leading_Edge_Cell_Lines.explode()
            .unique()
            .tolist()
            )
        cell_ids = X_train_df.index
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
        clf = LogisticRegression(
                l1_ratio=1.0,  # Equal mix of L1 (Lasso) and L2 (Ridge)
                C=C,         # Inverse regularization strength
                solver="liblinear",
                # tol=1e-3,
                max_iter=10000,
                random_state=42 + fold  # Vary random state per fold
            )
        clf.fit(F_augmented_df, y_union)

        val_leading_edge_score_tbl = compute_leading_edge_quantiles_vectorized(X_val,leading_edge_member_list)
        val_out_path_leading_edge_score_tbl = val_leading_edge_score_tbl.loc[:, out_path]

        val_F_augmented_df = augment_features_for_or_logic(val_out_path_leading_edge_score_tbl, tmp_res)
        oof_probs[val_idx] = clf.predict_proba(val_F_augmented_df)[:, 1]
    # --- OOF EVALUATION METRICS ---
        # Compute AUPRG on continuous predicted probabilities (not binary oof_preds)
        # Store auxiliary metrics as trial user attributes for later retrieval
        # TODO: Spearman correleation between proba and AUC as optimisation criteria
        val_criteria = 0
        return val_criteria

# %%


drug_id = 1372
tmp_drug_excess_mutation_count_tbl = src.mutation.gene_set_analysis.get_excess_mutation_count_matrix(drug_id,K_matrix,N_vector,dose_data_tbl,all_cells,all_genes)

tmp_drug_data_tbl = dose_data_tbl.query('DRUG_ID == @drug_id')
drug_name = tmp_drug_data_tbl.DRUG_NAME.iloc[0]

gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_excess_mutation_count_tbl,all_cells,all_genes,gene_set_to_use_dict)

shared_cell_id_list = list(set(tmp_drug_data_tbl.SANGER_MODEL_ID.unique().tolist()).intersection(gene_set_collection_excess_count_df.index.tolist())) 

tmp_drug_sensitivity_series = tmp_drug_data_tbl.query('SANGER_MODEL_ID in @shared_cell_id_list').set_index('SANGER_MODEL_ID').loc[shared_cell_id_list,'sensitivity_p']

tmp_drug_excess_mutation_count_tbl = gene_set_collection_excess_count_df.loc[shared_cell_id_list,:]
# %%
# --- EXECUTE OPTUNA STUDY ---
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(logit_objective, n_trials=150, timeout=1800)  # Adjust trials/timeout as needed

# %%
