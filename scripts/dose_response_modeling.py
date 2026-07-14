import pandas as pd
import bioframe as bf
import numpy as np
import matplotlib.pyplot as plt
import xlmhglite
from pathlib import Path
from scipy.stats import beta, gaussian_kde
from importlib import reload
import src.dose_response.detect_response
reload(src.dose_response.detect_response)
# %%
dose_fit_file = "/home/vipink/Documents/dose_response_workflow/data/GDSC2_fitted_dose_response_27Oct23.csv"


dose_coef_tbl = pd.read_csv(dose_fit_file,sep='\t')
dose_data_tbl = dose_coef_tbl.loc[:,['SANGER_MODEL_ID','CANCER_TYPE', 'DRUG_ID', 'DRUG_NAME','MIN_CONC', 'MAX_CONC','LN_IC50','AUC', 'RMSE']]

null_dose_data_tbl = dose_data_tbl.assign(LN_MAX_CONC = lambda df: np.log(df.MAX_CONC)).assign(inert = lambda df: df.LN_IC50.gt(df.LN_MAX_CONC)).query('inert')

# %%
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

plt.title("The 'shrunken' AUC null distribution landscape", fontsize=15)
plt.xlabel("AUC (Area Under the Curve)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.xlim(0.6, 1.05) # Focus on the high-AUC region where nulls live
plt.grid(axis='y', alpha=0.3)

plt.show()
# %%
dose_data_tbl = dose_data_tbl.merge(
        drug_beta_param_df[['DRUG_ID','alpha', 'beta']], 
        how='left')
dose_data_tbl = dose_data_tbl.assign(sensitivity_p = lambda df: beta.cdf(df['AUC'], df['alpha'], df['beta']))

# %%

dose_data_tbl = dose_data_tbl.assign(rank_score = lambda df: -np.log10(df.sensitivity_p + 1e-100),adjusted_auc_rank = lambda df: df.sensitivity_p.rank(pct=True)) # Avoid log(0)
# %%
tmp_ax = dose_data_tbl.plot.scatter(x='AUC',y='adjusted_auc_rank',c='alpha',s=1,alpha=0.2,logy=True)
plt.show()
