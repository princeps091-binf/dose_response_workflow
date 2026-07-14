from scipy.stats import beta, false_discovery_control
import pandas as pd
import numpy as np

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


