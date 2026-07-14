import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import reload
from scipy.stats import beta
import xlmhglite
import networkx as nx
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

all_wes_mutation_df = pd.concat([src.utils.io.get_vcf_summary_tbl(vcf_file) for vcf_file in vcf_file_list]).drop_duplicates()
# %%

gene_set_dict = src.utils.io.parse_gmt(gene_set_file)
Gene_Set_size_tbl = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).explode('Genes').Gene_Set.value_counts().reset_index().rename(columns={'count':'gene_count'})

sub_collection_list = ['REACTOME','KEGG','PID']
collection_to_use_list = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).assign(collection = lambda df: [i.split('_')[0] for i in df.Gene_Set]).query('collection in @sub_collection_list').Gene_Set.drop_duplicates().to_list()


# %%
Gene_Set_avg_gene_in_count_df = (
        pd.DataFrame.from_dict(list(gene_set_dict.items()))
        .rename(columns={0:'Gene_Set',1:'Genes'})
        .query('Gene_Set in @collection_to_use_list')
        .explode('Genes')
        .merge(all_wes_mutation_df.query('~(var_type == "silent")').loc[:,['sanger_model_id','gene']],how='left',left_on='Genes',right_on='gene')
        .assign(obs = lambda df: ~df.gene.isna())
        .groupby(['Gene_Set','sanger_model_id'])
        .agg(gene_in = ('obs','sum')).reset_index()
                                 )
Gene_Set_tot_count_tbl = Gene_Set_avg_gene_in_count_df.groupby('Gene_Set').agg(tot_count = ('gene_in','sum')).reset_index()
cell_id_number = Gene_Set_avg_gene_in_count_df.sanger_model_id.nunique() 

Gene_set_to_keep_list = Gene_Set_avg_gene_in_count_df.groupby('Gene_Set').agg(max_gene_in = ('gene_in','max')).query('max_gene_in > 15 and max_gene_in < 500').reset_index().Gene_Set.drop_duplicates().to_list()


Gene_set_to_keep_list = Gene_Set_avg_gene_in_count_df.groupby('Gene_Set').agg(max_gene_in = ('gene_in','max')).query('max_gene_in > 5 and max_gene_in < 500').reset_index().Gene_Set.drop_duplicates().to_list()

total_zscore_tbl = src.mutation.gene_set_analysis.parallel_zscore_estimation(Gene_Set_avg_gene_in_count_df.query('Gene_Set in @Gene_set_to_keep_list'),Gene_Set_tot_count_tbl,cell_id_number, n_cores=10)

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
# Highlight drug selection 

dose_data_tbl.assign(sensitivity_breadth = lambda df: df.sensitivity_p.lt(1e-3)).groupby(['DRUG_ID','DRUG_NAME']).agg(sensi_p = ('sensitivity_breadth','mean')).query('sensi_p > 0.1').sort_values('sensi_p')


dose_data_tbl.query('DRUG_NAME == "Trametinib"').DRUG_ID.unique()
# %%
#2 Lapatinib = 1558
#3 Vorinostat = 1012
# Gefitinib = 1010
#1 Trametinib = 1372 
tmp_drug = 1372

gene_set_drug_pair_list = [(path, tmp_drug) for path in total_zscore_tbl.Gene_Set.unique()]

#d_res_df = src.mutation.gene_set_analysis.parallel_mhgt(gene_set_drug_pair_list,dose_data_tbl,total_zscore_tbl,n_cores=8)


d_res_df = src.mutation.gene_set_analysis.parallel_mhgt_high_performance(gene_set_drug_pair_list,dose_data_tbl,total_zscore_tbl,n_cores=10)

# %%

# Identify outlyingly associated gene sets using p-value distribution elbow
from kneed import KneeLocator, find_shape
kneed_tbl = d_res_df.assign(x = lambda df: df.pvalue.rank(pct=True),y=lambda df:-np.log10(df.pvalue)).sort_values('x').loc[:,['Gene_Set','x','y']]


direction, curve = find_shape(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy())
kl = KneeLocator(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy(), curve=curve, direction=direction)
tmp_ax = kneed_tbl.plot(x='x',y='y')
tmp_ax.axvline(x=kl.knee, color='red', linestyle='--', linewidth=2, label='Threshold')
plt.show()
# %%
# Extract the pathways that constitute outlier association with the drug response
out_path = kneed_tbl.query('x <= @kl.knee').Gene_Set.to_list()
len(out_path)
# %%

# 1. Initialize your Analyzer with data blocks
analyzer = src.integration.gene_burden.MutationBurdenAnalyzer(
    d_res_df=d_res_df,
    total_zscore_tbl=total_zscore_tbl,
    dose_data_tbl=dose_data_tbl,
    all_wes_mutation_df=all_wes_mutation_df,
    gene_set_dict=gene_set_dict
)

# 2. Simultaneously fetch the Node DataFrame and the Network Edgelist
gene_dfs = []
edge_dfs = []

for idx in range(len(out_path)):
        print(idx/len(out_path))
        enriched_genes, pathway_edges = analyzer.analyze_pathway(pathway_id=out_path[idx], drug_id=tmp_drug)
        gene_dfs.append(enriched_genes)
        edge_dfs.append(pathway_edges)

enriched_burden_gene_df = pd.concat(gene_dfs)
edge_df = pd.concat(edge_dfs)

# %%

edge_df = (edge_df
 .merge(enriched_burden_gene_df.loc[:,['gene','pscore','Gene_Set','sanger_model_id']],
        left_on=['Gene_1','Gene_Set','sanger_model_id'],
        right_on=['gene','Gene_Set','sanger_model_id'])
 .rename(columns={'pscore':'Gene_1_weight'})
 .drop('gene',axis=1)
 .merge(enriched_burden_gene_df.loc[:,['gene','pscore','Gene_Set','sanger_model_id']],
        left_on=['Gene_2','Gene_Set','sanger_model_id'],
        right_on=['gene','Gene_Set','sanger_model_id'])
 .rename(columns={'pscore':'Gene_2_weight'})
 .drop('gene',axis=1)
)

g_list=[]
for idx in range(len(out_path)):
    candidate_edge_list_df = edge_df.query('Gene_Set == @out_path[@idx]').groupby(['Gene_1','Gene_2','Gene_Set','leading_edge_n']).agg(w = ('sanger_model_id','nunique'),vw1 = ('Gene_1_weight','mean'), vw2 = ('Gene_2_weight','mean')).query('w >1').reset_index().assign(weight = lambda df: df.w/df.leading_edge_n)
    G = nx.from_pandas_edgelist(
        candidate_edge_list_df, 
        source='Gene_1', 
        target='Gene_2', 
        edge_attr='weight',  # Stores this column as an edge attribute
        create_using=nx.Graph()        # Ensures it is Undirected
    )
    vertex_dict = pd.concat([candidate_edge_list_df.loc[:,['Gene_1','vw1']].drop_duplicates().rename(columns={'vw1':'vw','Gene_1':'Gene'}),candidate_edge_list_df.loc[:,['Gene_2','vw2']].drop_duplicates().rename(columns={'vw2':'vw','Gene_2':'Gene'})]).groupby('Gene').agg(vm = ('vw','mean')).to_dict()
    nx.set_node_attributes(G, vertex_dict['vm'], name='burden')
    g_list.append(G)


agg_G = src.integration.leading_edge.aggregate_gene_set_networks(g_list)

# %%
# Compute layout using Spectral Sorting
pos = src.integration.leading_edge.spectral_hilbert_layout(agg_G, gap_size=4)
centrality = nx.degree_centrality(agg_G)
node_sizes = [1 + (centrality[node] * 10000) for node in agg_G.nodes()]
# Plotting the result
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(agg_G, pos, node_size=node_sizes, node_color='#4D96FF')
nx.draw_networkx_edges(agg_G, pos, alpha=0.2, edge_color='black')
nx.draw_networkx_labels(agg_G, pos, font_size=10, font_color='grey')

plt.title("Spectral (Fiedler Vector) Hilbert Layout")
plt.axis('off')
plt.show()

# %%

nx.draw(agg_G)
plt.show()
# %%
cell_dfs = []
for tmp_path in out_path:
    z_thresh = d_res_df.query('Gene_Set == @tmp_path').z_thresh.values[0]
    cutoff = int(d_res_df.query('Gene_Set == @tmp_path').cut_off.values[0])
    # 2. Segment Cell Lines (Responsive vs Unresponsive)
    path_df = total_zscore_tbl.query('Gene_Set == @tmp_path')
    original_responders = path_df.query('z_score > @z_thresh').cell_id.to_list()
    drug_dose_df = dose_data_tbl.query('DRUG_ID == @tmp_drug').sort_values('sensitivity_p')
    leading_edge_cells = (
        drug_dose_df.iloc[:cutoff, :]
        .assign(in_pathway=lambda df: np.where(df.SANGER_MODEL_ID.isin(original_responders), 1, 0))
        .query('in_pathway > 0')
        .SANGER_MODEL_ID.to_list()
    )
    unresponsive_cells = drug_dose_df.iloc[cutoff:, :].SANGER_MODEL_ID.to_list()
    cell_dfs.append(pd.concat([pd.DataFrame({'sanger_model_id':leading_edge_cells})
     .assign(Gene_Set = tmp_path)
     .assign(kind = 'leading_edge'),
     pd.DataFrame({'sanger_model_id':unresponsive_cells})
     .assign(Gene_Set = tmp_path)
     .assign(kind = 'unresponsive'),
     ]))

# %%

cell_line_selection_tbl = pd.concat(cell_dfs).query('kind == "leading_edge"').sanger_model_id.value_counts().reset_index().rename(columns={'count':'lead_count'}).merge(pd.concat(cell_dfs).query('kind == "unresponsive"').sanger_model_id.value_counts().reset_index().rename(columns={'count':'resistant_count'}),how='outer',on='sanger_model_id').fillna(0)

# Contrast analysis between consistently leading and non-responsive cell lines as heatmap overlap
always_resistant_cell_lines_list = cell_line_selection_tbl.query('lead_count<1').sanger_model_id.to_list()
always_lead_cell_lines_list = cell_line_selection_tbl.query('resistant_count<1').sanger_model_id.to_list()
mostly_lead_cell_lines_list = (
cell_line_selection_tbl
.assign(tot_count = lambda df: df.lead_count + df.resistant_count)
.assign(lead_factor = lambda df: df.lead_count/df.resistant_count)
.assign(lead_ratio = lambda df: df.lead_count/df.tot_count)
.query('lead_factor >= 2')
.sanger_model_id.to_list()
)
tmp_drug_null_tot_mut = all_wes_mutation_df.query('sanger_model_id in @always_resistant_cell_lines_list').POS.count()

tmp_drug_pnull_tbl = all_wes_mutation_df.query('sanger_model_id in @always_resistant_cell_lines_list').groupby(['gene']).agg(mcount=('POS','count')).assign(pnull = lambda df: df.mcount/tmp_drug_null_tot_mut).reset_index()
tmp_cell_tbl = all_wes_mutation_df.query('sanger_model_id in @mostly_lead_cell_lines_list').groupby('gene').agg(obs = ('POS','count')).reset_index()
from scipy.stats import binom
tmp_cell_gene_burden_tbl = tmp_cell_tbl.assign(tot_count = lambda df: all_wes_mutation_df.query('sanger_model_id == @mostly_lead_cell_lines_list').POS.count()).merge(tmp_drug_pnull_tbl.loc[:,['gene','pnull']],how='left').assign(binomp=lambda df: binom.sf(df.obs - 1, df.tot_count, df.pnull)).query('gene in @agg_G.nodes')
print(tmp_cell_gene_burden_tbl.shape[0])
tmp_ax = tmp_cell_gene_burden_tbl.binomp.plot.kde()
plt.show()

# %%
active_nodes_tbl = tmp_cell_gene_burden_tbl.loc[:,['gene','binomp']].query('binomp < 0.5')
active_vals = active_nodes_tbl.binomp.to_numpy()
discretized_significance = np.zeros(active_nodes_tbl.shape[0])
# 2. Calculate Decile Boundaries (10%, 20%, ..., 90%)
decile_thresholds = np.percentile(-np.log10(active_vals), np.arange(10, 101, 10))
for idx in range(active_nodes_tbl.shape[0]):
    val = -np.log10(active_nodes_tbl.binomp.iloc[idx])
    if val <= -np.log10(0.5):
        discretized_significance[idx] = 0.0
    else:
        # Find which decile bucket the p-value falls into
        # np.searchsorted finds the index in [10%, 20%... 100%]
        bucket = np.searchsorted(decile_thresholds, val)
        # Map to a 0.1 - 1.0 scale
        discretized_significance[idx] = (bucket + 1) / 10.0

tmp_cell_gene_burden_tbl = tmp_cell_gene_burden_tbl.loc[:,['gene','binomp']].merge(active_nodes_tbl.assign(qscore=discretized_significance),how='left').fillna(0)

# %%

agg_G_node_attr_tbl = pd.DataFrame.from_dict(pos,orient='index',columns=['x','y']).reset_index().rename(columns={'index':'gene'}).merge(tmp_cell_gene_burden_tbl.loc[:,['gene','qscore']])


node_sizes = [tmp_cell_gene_burden_tbl.query('gene == @node').qscore.iloc[0] * 1e3 if tmp_cell_gene_burden_tbl.query('gene == @node').shape[0] > 0 else 10 for node in agg_G.nodes() ]


exi,eyi,ezi = src.integration.leading_edge.generate_edge_contour_matrices(pos,agg_G,pd.DataFrame(pos).iloc[0,:].max())

fig, ax = plt.subplots(figsize=(13, 12))  # Dark tech background
ax.set_facecolor('#090d16')
nx.draw_networkx_nodes(agg_G, pos, node_size=node_sizes, node_color='#4D96FF')
nx.draw_networkx_edges(agg_G, pos, alpha=0.5, edge_color='grey')
nx.draw_networkx_labels(agg_G, pos, font_size=10, font_color='black')
contour_filled = ax.contourf(exi, eyi, ezi, levels=20, cmap='plasma', alpha=0.8, zorder=1)
ax.set_xlim(-1, agg_G_node_attr_tbl.x.max() + 1)
ax.set_ylim(-1, agg_G_node_attr_tbl.y.max() + 1)
plt.title("Spectral (Fiedler Vector) Hilbert Layout")
plt.axis('off')
plt.show()

# %%

tmp_cell_gene_burden_tbl.sort_values("binomp").query('qscore > 0.7')

# %%

dose_data_tbl.query('DRUG_ID == @tmp_drug and SANGER_MODEL_ID in @mostly_lead_cell_lines_list').loc[:,['CANCER_TYPE','DRUG_NAME','AUC']]

# %%
dose_data_tbl.query('DRUG_ID == @tmp_drug and SANGER_MODEL_ID in @mostly_lead_cell_lines_list').AUC.mean()

dose_data_tbl.query('DRUG_ID == @tmp_drug and SANGER_MODEL_ID in @always_resistant_cell_lines_list').AUC.mean()

# %%
# Ladnscape as tool for determining single sample sensitivity
tot_resistant_cell_lines_mutation_count = (all_wes_mutation_df
 .query('sanger_model_id in @always_resistant_cell_lines_list')
 .POS
 .count())
resistant_mutation_pnull_tbl = (all_wes_mutation_df
 .query('sanger_model_id in @always_resistant_cell_lines_list')
 .query('gene in @active_nodes_tbl.gene.to_list()')
 .groupby('gene')
 .agg(m_count = ('POS','count'))
 .reset_index()
 .assign(p_null = lambda df: df.m_count/tot_resistant_cell_lines_mutation_count)
        )
# %%


tot_mutation_count_per_resistant_cell_line_tbl = (all_wes_mutation_df
                                        .query('sanger_model_id in @always_resistant_cell_lines_list')
                                        .groupby('sanger_model_id')
                                        .agg(k_trial = ('POS','count'))
                                        .reset_index()
)

tot_mutation_count_per_cell_line_tbl = (all_wes_mutation_df
                                        .groupby('sanger_model_id')
                                        .agg(k_trial = ('POS','count'))
                                        .reset_index()
)

# %%
resistant_cell_line_binomp_tbl = (
all_wes_mutation_df
 .query('sanger_model_id in @always_resistant_cell_lines_list')
 .query('gene in @active_nodes_tbl.gene.to_list()')
 .groupby(['sanger_model_id','gene'])
 .agg(mcount = ('POS','count'))
 .reset_index()
 .merge(tot_mutation_count_per_resistant_cell_line_tbl,how='left')
 .merge(resistant_mutation_pnull_tbl.loc[:,['gene','p_null']],how='left')
 .fillna(1e-5)
 .assign(binomp=lambda df: binom.sf(df.mcount - 1, df.k_trial, df.p_null))
 .assign(pscore = lambda df: -np.log10(df.binomp))
        )

# %%
gene_pscore_thresh = {}
for gene in active_nodes_tbl.gene.to_list():
    gene_bg_scores = resistant_cell_line_binomp_tbl.query('gene == @gene').pscore.sort_values().to_numpy()
    if len(gene_bg_scores) > 3 :
        x_indices = np.arange(len(gene_bg_scores))
        # Locate the elbow point along the population curve
        kneedle = KneeLocator(x_indices, gene_bg_scores, curve="convex", direction="increasing")
        if kneedle.knee_y is not None:
            theta_g = kneedle.knee_y
        else:
            theta_g = np.percentile(gene_bg_scores, 95) # High-stringency fallback
    else:
        theta_g = 0.0
# Confirm activation if it clears the baseline population elbow threshold
    gene_pscore_thresh[gene] = theta_g

# %%

trial_cell_line_binomp_tbl = (
all_wes_mutation_df
 .query('gene in @active_nodes_tbl.gene.to_list()')
 .groupby(['sanger_model_id','gene'])
 .agg(mcount = ('POS','count'))
 .reset_index()
 .merge(tot_mutation_count_per_cell_line_tbl,how='left')
 .merge(resistant_mutation_pnull_tbl.loc[:,['gene','p_null']],how='left')
 .fillna(1e-5)
 .assign(binomp=lambda df: binom.sf(df.mcount - 1, df.k_trial, df.p_null))
 .assign(pscore = lambda df: -np.log10(df.binomp))
 .merge(pd.DataFrame.from_dict(gene_pscore_thresh,orient='index').reset_index().rename(columns={'index':'gene',0:'pthresh'}))
       )
# %%
active_nodes_tbl.assign(pweight = lambda df: -np.log10(df.binomp)/sum(-np.log10(df.binomp)))
