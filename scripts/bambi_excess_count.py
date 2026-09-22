
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
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
total_exome_loads = all_wes_mutation_df.query('var_type != "silent"').groupby('sanger_model_id').size().rename('total_cell_muts')
base_matrix = all_wes_mutation_df.groupby(['sanger_model_id', 'gene']).size().unstack(fill_value=0)

all_cells = base_matrix.index.tolist()
all_genes = base_matrix.columns.tolist()

# Core numpy structures
K_matrix = base_matrix.values  # Shape: (n_cells, n_genes)
N_vector = total_exome_loads.reindex(all_cells).fillna(0).values.reshape(-1, 1) # Shape: (n_cells, 1)

# %%
res_df = []
for tmp_gene in base_matrix.columns:
        print(tmp_gene)
        gene_mutation_count_df = base_matrix.loc[:,[tmp_gene]].reset_index().rename(columns={tmp_gene:'gene_mut_count'}).merge(
        dose_data_tbl.loc[:,['SANGER_MODEL_ID','CANCER_TYPE']].rename(columns={'SANGER_MODEL_ID':'sanger_model_id'}).drop_duplicates(),on='sanger_model_id').merge(total_exome_loads.reset_index())
        # Demonstrate the tissue confounder effect
        # 2. TMB-Adjusted ANOVA (Log-Linear)
        # Tests if tissue bias persists AFTER controlling for overall sample TMB
        model_tmb = ols("gene_mut_count ~ total_cell_muts + C(CANCER_TYPE)", data=gene_mutation_count_df).fit()
        anova_tmb = sm.stats.anova_lm(model_tmb, typ=2)
        res_df.append(anova_tmb.loc[:,['PR(>F)']].assign(gene=tmp_gene))

# %%

pd.concat(res_df).reset_index().rename(columns={'index':'var'}).query('var == "C(CANCER_TYPE)"').loc[:,'PR(>F)'].plot.kde()
plt.show()

# %%

(base_matrix>0).sum(axis=0).reset_index().rename(columns={0:'mut_count'}).query('mut_count > 15')
# %%
drug_id = 1372
null_cell_lines_list = null_dose_data_tbl.query('DRUG_ID == @drug_id').SANGER_MODEL_ID.unique().tolist()

tmp_gene = 'RYR2'
gene_mutation_count_df = base_matrix.loc[:,[tmp_gene]].reset_index().rename(columns={tmp_gene:'gene_mut_count'}).query('sanger_model_id in @null_cell_lines_list').merge(
dose_data_tbl.loc[:,['SANGER_MODEL_ID','CANCER_TYPE']].rename(columns={'SANGER_MODEL_ID':'sanger_model_id'}).drop_duplicates(),on='sanger_model_id').merge(total_exome_loads.reset_index())

gene_mutation_count_df = gene_mutation_count_df.assign(
                log_total_muts = lambda df: np.log(df.total_cell_muts))

# %%
# Produce the null posterior for considered drug based on corresponding resistant cell lines
# Enable inference on unseen groups when predicting

import bambi as bmb
# 2. Specify the Bambi formula
# 'offset()' fixes the coefficient of log_total_muts to 1.0
# '(1 | tissue_type)' adds a random intercept for tissue of origin
formula = "gene_mut_count ~ 1 + offset(log_total_muts) + (1 | CANCER_TYPE)"

#tmp_drug_excess_mutation_count_tbl.iloc[train_idx] 3. Fit the Negative Binomial model to handle overdispersion
model = bmb.Model(
    formula=formula,
    data=gene_mutation_count_df,
    family="negativebinomial",  # Use 'poisson' if data shows no overdispersion
    link="log"
)

# %%
# 4. Sample from posterior
results = model.fit(
    draws=2000,
    tune=1000,
    chains=4,
    target_accept=0.95,
    random_seed=42
)
# %%
import arviz as az

az.plot_trace(results)
plt.show()

# %%

az.plot_posterior(results)
plt.show()
# %%
idata = model.predict(results,kind='response',inplace=False)

# %%
gene_mutation_for_pred_df = base_matrix.loc[:,[tmp_gene]].reset_index().rename(columns={tmp_gene:'gene_mut_count'}).merge(
dose_data_tbl.loc[:,['SANGER_MODEL_ID','CANCER_TYPE']].rename(columns={'SANGER_MODEL_ID':'sanger_model_id'}).drop_duplicates(),on='sanger_model_id').merge(total_exome_loads.reset_index())

gene_mutation_for_pred_df = gene_mutation_for_pred_df.assign(
                log_total_muts = lambda df: np.log(df.total_cell_muts))

preds = model.predict(
    results,
    kind="response",
    data=gene_mutation_for_pred_df,
    sample_new_groups=True,
    inplace=False
)

# %%
preds.posterior_predictive['gene_mut_count'][0,:,0]

preds.observed_data['gene_mut_count']
