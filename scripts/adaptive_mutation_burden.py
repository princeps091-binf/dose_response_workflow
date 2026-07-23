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
drug_id = 1010
drug_name = dose_data_tbl.query('DRUG_ID == @drug_id').DRUG_NAME.iloc[0]
tmp_drug_excess_mutation_count_tbl = src.mutation.gene_set_analysis.get_excess_mutation_count_matrix(drug_id,K_matrix,N_vector,dose_data_tbl,all_cells,all_genes)


gene_set_collection_excess_count_df = src.mutation.gene_set_analysis.compute_all_pathway_burdens_vectorized(tmp_drug_excess_mutation_count_tbl,all_cells,all_genes,gene_set_to_use_dict)


tmp_res = src.mutation.gene_set_analysis.run_high_throughput_parallel_xlmhg(
    pathway_burden_df = gene_set_collection_excess_count_df,   
    drug_sensitivity_df = dose_data_tbl.query('DRUG_ID == @drug_id'), 
    n_burden_steps = 20,
    auc_col = 'sensitivity_p',
    sanger_id_col = 'SANGER_MODEL_ID',
    n_jobs = 8  # Use all available CPU cores
)

# %%
from kneed import KneeLocator, find_shape
kneed_tbl = tmp_res.assign(x = lambda df: df.Min_mHG_P_Value.rank(pct=True),y=lambda df:-np.log10(df.Min_mHG_P_Value)).sort_values('x').loc[:,['Pathway_Name','x','y']]


direction, curve = find_shape(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy())
kl = KneeLocator(kneed_tbl.x.to_numpy(), kneed_tbl.y.to_numpy(), curve=curve, direction=direction)

# Extract the pathways that constitute outlier association with the drug response
out_path = kneed_tbl.query('x <= @kl.knee').Pathway_Name.to_list()

# tmp_ax = kneed_tbl.plot(x='x',y='y')
# tmp_ax.axvline(x=kl.knee, color='red', linestyle='--', linewidth=2, label='Threshold')
# plt.show()
# %%

leading_enrich_stat_tbl = src.integration.leading_edge.produce_cell_type_enrichment_tbl(drug_id,out_path,tmp_res,dose_data_tbl)

# %%
import plotly.express as px
fig = px.treemap(
        leading_enrich_stat_tbl,
        path=[px.Constant("All Leading Edge Cells"), 'CANCER_TYPE'],
        values='ncl',
        color='z_score',
        color_continuous_scale='RdBu_r', # Red = Enriched, Blue = Depleted
        color_continuous_midpoint=0.0,
        hover_data={
            'ncl': True,
            'bg_n': True,
            'expected_k': ':.2f',
            'z_score': ':.2f',
            'pval': ':.2e'
        },
        labels={
            'ncl': 'Leading Edge Count',
            'bg_n': 'Total Library Count',
            'z_score': 'Enrichment Z-Score',
            'pval': 'p-value'
        },
        title=f"Cell-Type Composition & Enrichment for {drug_name} Leading Edge)"
    )

fig.update_traces(
    hovertemplate="<b>%{label}</b><br>" +
                  "Observed Count: %{value}<br>" +
                  "Enrichment Z-score: %{color:.2f}<br>" +
                  "<extra></extra>"
)
fig.update_layout(
    margin=dict(t=50, l=10, r=10, b=10),
    coloraxis_colorbar=dict(title="Enrichment<br>Z-Score")
)

fig.write_html(f'img/treemap_{drug_name}.html')
# %%
agg_edge_df_list = []
agg_node_df_list = []

for tmp_gene_set_name in out_path:
    print(tmp_gene_set_name)
    tmp_gene_set = gene_set_to_use_dict[tmp_gene_set_name]
    tmp_gene_set_leading_edge_list = tmp_res.query('Pathway_Name == @tmp_gene_set_name').Leading_Edge_Cell_Lines.iloc[0]
    node_df, edge_df = src.integration.leading_edge.construct_leading_edge_network(tmp_gene_set,tmp_gene_set_leading_edge_list,tmp_drug_excess_mutation_count_tbl)
    agg_edge_df_list.append(edge_df.assign(Pathway_Name = tmp_gene_set_name))
    agg_node_df_list.append(node_df.assign(Pathway_Name = tmp_gene_set_name))


# %%

obs_paths_list = pd.concat(agg_node_df_list).Pathway_Name.unique()
agg_node_df, agg_edge_df = src.integration.leading_edge.aggregate_pathway_networks_probabilistic_product(agg_node_df_list,agg_edge_df_list,obs_paths_list)

tmp_ax = agg_edge_df.assign(pr = lambda df: df.Consolidated_Edge_Weight.rank(pct=True)).plot(x='pr',y='Consolidated_Edge_Weight')
plt.show()


# %%
agg_G =  nx.from_pandas_edgelist(
        agg_edge_df.loc[:,['Source','Target','Consolidated_Edge_Weight']].rename(columns={'Consolidated_Edge_Weight':'weight'}), 
    source='Source', 
    target='Target', 
    edge_attr='weight' 
)
pos = src.integration.leading_edge.spectral_hilbert_layout(agg_G, gap_size=4)

# Plotting the result

exi,eyi,ezi = src.integration.leading_edge.generate_edge_contour_matrices(pos,agg_G,pd.DataFrame(pos).iloc[0,:].max(),resolution=500)

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
# interactive form for this visualisation using plotly

src.integration.leading_edge.create_interactive_network_explorer(pos,agg_node_df,exi,eyi,ezi,output_html_path=f'./img/{drug_name}_c2_network.html')

