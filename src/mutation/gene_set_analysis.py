import numpy as np
from sklearn.linear_model import HuberRegressor
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import TweedieRegressor
import xlmhglite
from scipy.stats import binom
from joblib import Parallel, delayed

def get_mutation_burden_zscore(mutation_count_tbl):
    x_baseline = np.log10(mutation_count_tbl.avg_gene.to_numpy()).reshape(-1, 1)
    y_observed = np.log10(mutation_count_tbl.gene_in.to_numpy())
    # For Tweedie we only log the baseline not the observedmutated gene counts
    #x_baseline = np.log10(mutation_count_tbl.avg_gene.to_numpy()).reshape(-1, 1)
    #y_observed = mutation_count_tbl.gene_in.to_numpy()
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),HuberRegressor(epsilon=1.35))
    #model = TweedieRegressor(power=1.1, link='log')
    #model = HuberRegressor(epsilon=1.35)
    model.fit(x_baseline, y_observed)
    # 4. Predict the "Expected" mutation count for this specific TMB
    y_predicted = model.predict(x_baseline)
    # 5. Calculate Standardized Residuals (Z-scores)
    residuals = y_observed - y_predicted
    # Scale by the Median Absolute Deviation (MAD) for robustness
    # This prevents hyper-mutated genes from inflating the standard deviation
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = 1.4826 * mad # Consistency constant for normal distribution
    if sigma == 0: # Handle cases with very low mutation counts
        z_scores = residuals
    else:
        z_scores = residuals / sigma
    
    return  mutation_count_tbl.assign(z_score=z_scores,predicted_count = y_predicted)

def get_cell_line_pathway_mutation_enrichment_tbl(Gene_Set_count_tbl, cell_line_id):
    
    tmp_cell_model_count_tbl = Gene_Set_count_tbl.query('sanger_model_id == @cell_line_id').loc[:,['Gene_Set','gene_in']].merge(Gene_Set_count_tbl.query('sanger_model_id != @cell_line_id').loc[:,['Gene_Set','gene_in']].groupby(['Gene_Set']).agg(avg_gene = ('gene_in','mean')).reset_index())
    
    tmp_cell_model_count_tbl = get_mutation_burden_zscore(tmp_cell_model_count_tbl)
    
    return tmp_cell_model_count_tbl.assign(cell_id = cell_line_id)

from multiprocessing import Pool, get_context
# 1. We define a global 'initializer' to make the big matrix accessible
# to all workers without passing it as a repeated argument.
def init_worker(shared_count_tbl,gene_set_tot_count_tbl,n_cell_id):
    global global_count_tbl, global_sums, global_n
    global_count_tbl = shared_count_tbl
    global_sums = gene_set_tot_count_tbl
    global_n = n_cell_id
    

def _get_cell_line_pathway_mutation_enrichment_tbl_p(cell_line_id):
    

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

def get_hgmt_result_tbl(pathway_id,drug_id,path_zscore_tbl,dose_data_tbl,z_thresh):
    tmp_path_df = path_zscore_tbl.query('Gene_Set == @pathway_id')
    tmp_path_cell_id_array = tmp_path_df.query('z_score >= @z_thresh').cell_id.unique()
    cl_ranked_array = dose_data_tbl.query('DRUG_ID == @drug_id').assign(in_pathway = lambda df: np.where(df.SANGER_MODEL_ID.isin(tmp_path_cell_id_array),1,0)).sort_values('sensitivity_p').in_pathway.to_numpy()
    xstat, cutoff, pval = xlmhglite.xlmhg_test(cl_ranked_array, X=2, L=len(cl_ranked_array))
    return pd.DataFrame({'Gene_Set':[pathway_id],'DRUG_ID':[drug_id],'stat':[xstat],'cut_off':[cutoff],'pvalue':[pval],'z_thresh':[z_thresh]})


def get_optimal_hgmt_res(tmp_path,tmp_drug,total_zscore_tbl,dose_data_tbl):
    tmp_zarray = total_zscore_tbl.query('Gene_Set == @tmp_path').query('z_score > 0').z_score.to_numpy()
    z_tresh_array = np.arange(tmp_zarray.min(),tmp_zarray.max(),0.05)
    tmp_res_tbl = pd.concat([get_hgmt_result_tbl(tmp_path,tmp_drug,total_zscore_tbl,dose_data_tbl,tmp_z) for tmp_z in z_tresh_array])
    return tmp_res_tbl.sort_values('pvalue').head(1)


def mhgt_init_worker(dose_data_df,mutation_zscore_df):
    global global_dose, global_zscore
    global_dose = dose_data_df
    global_zscore = mutation_zscore_df
    

def _get_hgmt_result_tbl_p(pathway_id,drug_id,z_thresh):
    tmp_path_df = global_zscore.query('Gene_Set == @pathway_id')
    tmp_path_cell_id_array = tmp_path_df.query('z_score >= @z_thresh').cell_id.unique()
    cl_ranked_array = global_dose.query('DRUG_ID == @drug_id').assign(in_pathway = lambda df: np.where(df.SANGER_MODEL_ID.isin(tmp_path_cell_id_array),1,0)).sort_values('sensitivity_p').in_pathway.to_numpy()
    xstat, cutoff, pval = xlmhglite.xlmhg_test(cl_ranked_array, X=2, L=len(cl_ranked_array))
    return pd.DataFrame({'Gene_Set':[pathway_id],'DRUG_ID':[drug_id],'stat':[xstat],'cut_off':[cutoff],'pvalue':[pval],'z_thresh':[z_thresh]})

def _get_pathway_drug_optim_mhgt_p(pathway_id,drug_id):
    tmp_zarray = global_zscore.query('Gene_Set == @pathway_id').query('z_score > 0').z_score.to_numpy()
    z_tresh_array = np.arange(tmp_zarray.min(),tmp_zarray.max(),0.05)
    tmp_res_tbl = pd.concat([_get_hgmt_result_tbl_p(pathway_id,drug_id,tmp_z) for tmp_z in z_tresh_array])
    return tmp_res_tbl.sort_values('pvalue').head(1)

def parallel_mhgt(pathway_drug_pair,dose_data_df,mutation_zscore_df,n_cores=10):
    # Create the Pool
    # 'initializer' runs once for every worker when they start
    with Pool(processes=n_cores, 
              initializer=mhgt_init_worker, 
              initargs=(dose_data_df,mutation_zscore_df)) as pool:
        
        # starmap() will distribute the work and collect results in order
        results = pool.starmap(_get_pathway_drug_optim_mhgt_p, pathway_drug_pair, chunksize= 20)
        
    # Reconstruct the DataFrame
    res_df = pd.concat(results)
    return res_df

#--------
# Optimised parallelisation avoiding pandas dataframe in worker nodes

def mhgt_dict_init_worker(dose_data_dict,mutation_zscore_dict):
    global global_dose, global_zscore
    global_dose = dose_data_dict
    global_zscore = mutation_zscore_dict
 
def compute_single_pair_worker(task_args):
    """
    Accepts a single tuple of identifiers. Uses zero-copy global lookups,
    vectorized NumPy arrays, and executes at pure low-level C speeds.
    """
    pathway_id, drug_id = task_args
    
    # Fast O(1) dictionary lookups bypassing all Pandas overhead
    if pathway_id not in global_zscore or drug_id not in global_dose:
        return None
        
    # Extract pre-sorted model IDs and their corresponding p-value ranks
    drug_cl_models, drug_sensitivity_p = global_dose[drug_id]
    pathway_cell_ids, pathway_z_scores = global_zscore[pathway_id]
    
    # Filter out non-positive z-scores upfront
    valid_mask = pathway_z_scores > 0
    if not np.any(valid_mask):
        return None
    
    valid_z = pathway_z_scores[valid_mask]
    valid_cells = pathway_cell_ids[valid_mask]
    
    # Generate thresholds dynamically using NumPy vector operations
    z_thresh_array = np.arange(valid_z.min(), valid_z.max(), 0.05)
    
    best_pvalue = float('inf')
    best_record = None
    
    # Fast inner loop running on raw primitives
    for z_thresh in z_thresh_array:
        # Vectorized equivalent of global_zscore.query('z_score >= @z_thresh')
        active_cells = set(valid_cells[valid_z >= z_thresh])
        
        # Binary array mapping: Is the drug-tested cell line present in our active pathway?
        # Vectorized replacement for .isin() and np.where()
        cl_ranked_array = np.array([1 if cid in active_cells else 0 for cid in drug_cl_models], dtype=np.int8)
        
        # Compute the core mHG test statistics
        L_len = len(cl_ranked_array)
        xstat, cutoff, pval = xlmhglite.xlmhg_test(cl_ranked_array, X=2, L=L_len)
        
        # Track the minimum p-value on the fly (replaces expensive pd.concat + sort)
        if pval < best_pvalue:
            best_pvalue = pval
            best_record = (pathway_id, drug_id, xstat, cutoff, pval, z_thresh)
            
    return best_record

def parallel_mhgt_high_performance(pathway_drug_pairs, dose_data_df, mutation_zscore_df, n_cores=10):
    """
    The master pipeline coordinator. Compiles indexes in the parent process 
    and handles dynamic load balancing across workers.
    """
    print("Pre-indexing pharmacogenomic data frames...")
    
    # Pre-index Drugs: Group cell lines per drug sorted by sensitivity_p
    drug_map = {}
    for drug_id, group in dose_data_df.sort_values('sensitivity_p').groupby('DRUG_ID'):
        drug_map[drug_id] = (group['SANGER_MODEL_ID'].to_numpy(), group['sensitivity_p'].to_numpy())
        
    # Pre-index Pathways: Cache cell lines and z-scores as aligned arrays
    pathway_map = {}
    for pathway_id, group in mutation_zscore_df.groupby('Gene_Set'):
        pathway_map[pathway_id] = (group['cell_id'].to_numpy(), group['z_score'].to_numpy())
        
    ctx = get_context('fork')
    print(f"Launching pool with {n_cores} cores using dynamic scheduling...")
    
    with ctx.Pool(processes=n_cores, 
                  initializer=mhgt_dict_init_worker, 
                  initargs=(drug_map, pathway_map)) as pool:
                  
        # We drop starmap and use imap_unordered with chunksize=1.
        # Passing simple tuples takes nanoseconds, meaning chunksize=1 will not create lag.
        # This completely guarantees that no core sits idle.
        raw_results = pool.imap_unordered(compute_single_pair_worker, pathway_drug_pairs, chunksize=1)
        
        # Collect valid non-empty arrays
        clean_results = [r for r in raw_results if r is not None]
        
    # Reconstruct the unified final result DataFrame in one pass
    res_df = pd.DataFrame(clean_results, columns=['Gene_Set', 'DRUG_ID', 'stat', 'cut_off', 'pvalue', 'z_thresh'])
    return res_df

#-------------------------------
# Excess mutation based enrichment analysis

def get_excess_mutation_count_matrix(drug_id,gene_agg_count_per_cell_matrix,agg_cell_count_matrix,dose_data_tbl,all_cells,all_genes):
    print("Step 1: Vectorizing the drug-specific resistance masks...")
    # Label cell line resistance per drug via boolean query
    dose_data_tbl = dose_data_tbl.assign(LN_MAX_CONC = lambda df: np.log(df.MAX_CONC)).assign(is_resistant = lambda df: df.LN_IC50.gt(df.LN_MAX_CONC).astype(int))
    resistant_cells = set(dose_data_tbl.query('DRUG_ID == @drug_id and is_resistant > 0').SANGER_MODEL_ID.to_list())
    is_in_pool = np.array([1 if cell in resistant_cells else 0 for cell in all_cells]).reshape(-1, 1) # Shape: (n_cells, 1)
    total_resistant_count = np.sum(is_in_pool)
    if total_resistant_count == 0:
        print(f"Warning: Zero resistant cell lines found for drug '{drug_id}'. Using global background.")
        # Fall back to using all cell lines as the background pool to prevent division by zero
        is_in_pool = np.ones((len(all_cells), 1))
    
    # 3. Compute pool grand totals across the resistant cohort via matrix operations
    grand_k_drug = np.sum(gene_agg_count_per_cell_matrix * is_in_pool, axis=0) # Shape: (n_genes,)
    grand_n_drug = np.sum(agg_cell_count_matrix * is_in_pool)        # Scalar sum of total exome muts in pool
    
    epsilon = (1/grand_n_drug) * 1e-1
    # 4. Vectorized Leave-One-Out (LOO) Adjustment
    # Only subtract the cell's individual footprint if it is part of the resistant pool
    loo_k = np.maximum(0, grand_k_drug - (is_in_pool * gene_agg_count_per_cell_matrix))
    loo_n = np.maximum(1, grand_n_drug - (is_in_pool * agg_cell_count_matrix))
    
    # Calibrate expected background allocation rates
    p_null_matrix = (loo_k + epsilon) / (loo_n + (epsilon * 10))
    binom_p_matrix = binom.sf(gene_agg_count_per_cell_matrix - 1, agg_cell_count_matrix, p_null_matrix)
    binom_p_matrix = np.nan_to_num(binom_p_matrix, nan=1.0)
    excess_score_matrix = np.maximum(0.0, 1.0 - (2.0 * binom_p_matrix))
    
    drug_df = pd.DataFrame(excess_score_matrix, index=all_cells, columns=all_genes)
    stacked_series = drug_df.stack()
    stacked_series = stacked_series[stacked_series > 0.0]  # Drop structural zeros to compress memory
    flat_df = stacked_series.reset_index()
    flat_df.columns = ['sanger_model_id', 'gene', 'excess_mutation_count']
    
    if not stacked_series.empty:
        flat_df = stacked_series.reset_index()
        flat_df.columns = ['sanger_model_id', 'gene', 'excess_mutation_count']
        return flat_df
    else:
        return pd.DataFrame(columns=['sanger_model_id', 'gene', 'excess_mutation_count'])


def compute_all_pathway_burdens_vectorized(
    single_drug_excess_df: pd.DataFrame,  # Long-form output of your single-drug engine
    all_cells: list,                      # Reference master cell list (base_matrix.index)
    all_genes: list,                      # Reference master gene list (base_matrix.columns)
    gene_set_collection: dict             # {pathway_name: [list_of_genes]}
) -> pd.DataFrame:
    """
    Computes aggregate excess mutation burdens across an entire collection of gene sets
    simultaneously utilizing vectorized matrix-matrix dot products.
    
    Returns:
    --------
    pd.DataFrame
        Wide matrix of shape (n_cells, n_pathways) tracking continuous aggregate burden.
    """
    # 1. Unpack long-form drug excess signals back into an aligned dense numpy array
    # We map back to a fixed 2D grid so we can perform matrix multiplication
    print("Step 1: Reconstructing aligned drug feature matrix...")
    X_matrix = np.zeros((len(all_cells), len(all_genes)))
    
    if not single_drug_excess_df.empty:
        # Create indexing maps for lightning lookups
        cell_to_idx = {cell: i for i, cell in enumerate(all_cells)}
        gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
        
        # Extract row/column coordinate maps
        row_indices = single_drug_excess_df['sanger_model_id'].map(cell_to_idx).values
        col_indices = single_drug_excess_df['gene'].map(gene_to_idx).values
        values = single_drug_excess_df['excess_mutation_count'].values
        
        # Populate the matrix grid using multi-index advanced numpy arrays
        X_matrix[row_indices, col_indices] = values
    
    print("Step 2: Compiling the Gene-by-Pathway structural weight mapping...")
    # Extract structural pathway properties
    pathway_names = list(gene_set_collection.keys())
    gene_to_idx_global = {gene: i for i, gene in enumerate(all_genes)}
    
    # Pre-allocate binary footprint transformation array: Shape (n_genes, n_pathways)
    W_matrix = np.zeros((len(all_genes), len(pathway_names)))
    
    for p_idx, path_name in enumerate(pathway_names):
        # Isolate valid genes present in our historical exome footprint
        valid_pathway_genes = [g for g in gene_set_collection[path_name] if g in gene_to_idx_global]
        if valid_pathway_genes:
            target_gene_indices = [gene_to_idx_global[g] for g in valid_pathway_genes]
            # Flag these specific genes as components of the pathway row vector
            W_matrix[target_gene_indices, p_idx] = 1.0
    
    print("Step 3: Executing BLAS matrix-matrix dot product for aggregate burdens...")
    # Matrix Multiplication: (n_cells, n_genes) x (n_genes, n_pathways) -> (n_cells, n_pathways)
    # This automatically sums up the excess mutations for every gene inside each pathway
    pathway_burden_array = np.dot(X_matrix, W_matrix)
    
    # Wrap into a clean production DataFrame
    pathway_burden_df = pd.DataFrame(
        pathway_burden_array, 
        index=all_cells, 
        columns=pathway_names
    )
    pathway_burden_df.index.name = 'sanger_model_id'
    
    print(f"Success. Computed aggregated burdens across {len(pathway_names)} pathways for all cell lines.")
    return pathway_burden_df



def _process_single_pathway_worker(path_name, scores, sorted_cells, X_param, L_param, n_burden_steps, N):
    """
    Worker function executing the 2D sweep for a single pathway.
    Operates on a read-only slice of memory. Allocates its own local DP table.
    """
    positive_scores = scores[scores > 0.0]
    if len(positive_scores) < X_param:
        return None
        
    # Allocate a thread-local C-level DP table cache
    local_cache_table = np.empty((N + 1, N + 1), dtype=np.longdouble)
    
    max_safe_quantile = 1.0 - (X_param / N)
    min_safe_quantile = 1.0 - (len(positive_scores) / N)
    min_safe_quantile = min(min_safe_quantile, max_safe_quantile - 0.05)
    
    quantile_grid = np.linspace(min_safe_quantile, max_safe_quantile, n_burden_steps)
    tau_grid = np.unique(np.quantile(scores, quantile_grid))
    
    best_p_val = 1.0
    optimal_tau = 0.0
    optimal_cutoff = 0
    best_stat = 0.0
    
    for tau in tau_grid:
        v_vector = (scores >= tau).astype(np.int64)
        total_hits = np.sum(v_vector)
        
        if total_hits < X_param or total_hits >= N:
            continue
            
        stat, cutoff, pval = xlmhglite.xlmhg_test(
            v_vector, 
            X=X_param, 
            L=L_param, 
            table=local_cache_table
        )
        
        if pval < best_p_val:
            best_p_val = pval
            optimal_tau = tau
            optimal_cutoff = cutoff
            best_stat = stat
            
    if best_p_val < 1.0:
# --- Leading Edge Extraction Block ---
        # 1. Slice cell lines and their scores up to the optimal sensitivity index
        sensitive_cohort_cells = sorted_cells[:optimal_cutoff]
        sensitive_cohort_scores = scores[:optimal_cutoff]
        
        # 2. Isolate those in the sensitive cohort that pass the optimal mutation burden
        leading_edge_mask = sensitive_cohort_scores >= optimal_tau
        leading_edge_cells = sensitive_cohort_cells[leading_edge_mask]
        
        # 3. Convert to list format for clean embedding within the Pandas DataFrame cell
        leading_edge_list = list(leading_edge_cells)
        return {
            'Pathway_Name': path_name,
            'Min_mHG_P_Value': best_p_val,
            'Neg_Log_mHG_P': -np.log10(best_p_val + 1e-15),
            'Optimal_Burden_Threshold_Tau': optimal_tau,
            'Optimal_Sensitivity_Cutoff_Idx': optimal_cutoff,
            'Test_Statistic': best_stat,
            'Leading_Edge_Cell_Lines': leading_edge_list
        }
    return None

def run_high_throughput_parallel_xlmhg(
    pathway_burden_df: pd.DataFrame,   
    drug_sensitivity_df: pd.DataFrame, 
    n_burden_steps: int = 20,
    auc_col: str = 'AUC',
    sanger_id_col: str = 'sanger_model_id',
    n_jobs: int = -1  # Use all available CPU cores
) -> pd.DataFrame:
    """
    Parallelized 2D joint optimization screen utilizing zero-copy memory views.
    """
    print("Step 1: Aligning global read-only structures...")
    sorted_sens = drug_sensitivity_df.sort_values(by=auc_col, ascending=True).reset_index(drop=True)
    # CRITICAL: Keep as a fast-slicing NumPy array of strings/objects for the worker
    sorted_cells = np.array(sorted_sens[sanger_id_col].tolist(), dtype=object)
    # Isolate underlying dense NumPy arrays for raw pointer extraction
    aligned_burden = pathway_burden_df.reindex(sorted_cells).fillna(0.0)
    # CRITICAL FOR MULTI-PROCESSING: Ensure memory layout is continuous and C-aligned
    # This allows the operating system to share it perfectly via copy-on-write pointers
    K_matrix = np.ascontiguousarray(aligned_burden.values, dtype=np.float64)  # Shape: (n_cells, n_pathways)
    
    N = len(sorted_cells)
    pathway_names = aligned_burden.columns.tolist()
    
    X_param = 1
    L_param = int(N * 0.50)
    
    print(f"Step 2: Spawning zero-copy worker pool across {n_jobs} cores...")
    # Parallel map operation over columns. 
    # K_matrix[:, p_idx] generates a zero-copy memory view (slice), NOT a data duplicate.
    raw_results = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes='1M')(
        delayed(_process_single_pathway_worker)(
            path_name=pathway_names[p_idx],
            scores=K_matrix[:, p_idx],  # Read-only memory view slice
            sorted_cells= sorted_cells,
            X_param=X_param,
            L_param=L_param,
            n_burden_steps=n_burden_steps,
            N=N
        ) for p_idx in range(len(pathway_names))
    )
    
    # Clean up and filter out empty skipped pathways
    results_records = [r for r in raw_results if r is not None]
    
    summary_df = pd.DataFrame(results_records).sort_values(by='Neg_Log_mHG_P', ascending=False).reset_index(drop=True)
    print(f"Success. Parallel engine evaluated {len(summary_df)} active gene sets with zero data duplication.")
    return summary_df

