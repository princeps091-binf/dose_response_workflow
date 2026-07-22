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
drug_id = 1558

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
tmp_ax = kneed_tbl.plot(x='x',y='y')
tmp_ax.axvline(x=kl.knee, color='red', linestyle='--', linewidth=2, label='Threshold')
plt.show()
# %%
# Extract the pathways that constitute outlier association with the drug response
out_path = kneed_tbl.query('x <= @kl.knee').Pathway_Name.to_list()
# %%

leading_count_tbl = tmp_res.query("Pathway_Name in @out_path").loc[:,['Leading_Edge_Cell_Lines']].explode('Leading_Edge_Cell_Lines').drop_duplicates().merge(dose_data_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID','CANCER_TYPE']],left_on='Leading_Edge_Cell_Lines',right_on='SANGER_MODEL_ID').groupby('CANCER_TYPE').agg(ncl = ('SANGER_MODEL_ID','nunique')).reset_index()

null_tbl = dose_data_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID','CANCER_TYPE']].groupby('CANCER_TYPE').agg(bg_n = ('SANGER_MODEL_ID','nunique')).reset_index()

N_total = dose_data_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID']].drop_duplicates().shape[0]
N_le = tmp_res.query("Pathway_Name in @out_path").loc[:,['Leading_Edge_Cell_Lines']].explode('Leading_Edge_Cell_Lines').drop_duplicates().shape[0]
# %%
leading_enrich_stat_tbl = (
leading_count_tbl.merge(null_tbl)
.assign(
        pval = lambda df: hypergeom.sf(df.ncl -1,M=N_total,n=df.bg_n,N=N_le),
        expected_k = lambda df: hypergeom.stats(M=N_total,n=df.bg_n,N=N_le,moments='m'),
        variance_k = lambda df: hypergeom.stats(M=N_total,n=df.bg_n,N=N_le,moments='v'))
.assign(z_score =lambda df: (df.ncl - df.expected_k) / np.sqrt(df.variance_k))
)

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
        title=f"Cell-Type Composition & Enrichment for {drug_id} Leading Edge (n={N_le})"
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

fig.write_html(f'treemap_{drug_id}.html')

# %%
def construct_leading_edge_network(
    pathway_genes: list,
    leading_edge_cells: list,
    excess_mutation_df: pd.DataFrame # Index: sanger_model_id, Columns: genes
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs node and edge tables for the leading edge gene network.
    
    excess_mutation_df should be the pivot/matrix form of your excess scores 
    filtered/reindexed to contain only (leading_edge_cells x pathway_genes).
    """
    # 1. Align and slice the matrix to only include our active sub-space
    # Fill NaN with 0.0 to ensure safe math
    sub_matrix_df = (excess_mutation_df
                  .query('sanger_model_id in @leading_edge_cells and gene in @pathway_genes')
                  .pivot_table(index = 'sanger_model_id',columns='gene')
                  .fillna(0.0)
                  )
    obs_genes = [t[1] for t in sub_matrix_df.columns]
    sub_matrix = sub_matrix_df.values
    #print(obs_genes)
    n_cells, n_genes = sub_matrix.shape
    if n_cells == 0 or n_genes == 0:
        return pd.DataFrame(), pd.DataFrame()
    # Total pathway burden per cell line (Sum across columns)
    # Shape: (n_cells, 1) to allow broadcasting
    cell_pathway_totals = np.sum(sub_matrix, axis=1, keepdims=True)
    # Avoid division by zero for cell lines with zero overall burden
    cell_pathway_totals[cell_pathway_totals == 0.0] = 1.0 
    
    # --- Compute Node Weights ---
    # Prevalence: % of cells where score > 0
    node_prevalence = np.mean(sub_matrix > 0.0, axis=0)
    # Intensity: Average excess score over the leading edge
    node_intensity = np.mean(sub_matrix, axis=0)
    node_records = []
    for g_idx, gene in enumerate(obs_genes):
        node_records.append({
            'Gene': gene,
            'Weight_Prevalence': node_prevalence[g_idx],
            'Weight_Intensity': node_intensity[g_idx]
        })
    nodes_df = pd.DataFrame(node_records)
    
    # --- Compute Edge Weights ---
    edge_accumulator = []
    
    # Generate all unique gene pairs within the pathway
    for i, j in itertools.combinations(range(n_genes), 2):
        gene_i = obs_genes[i]
        gene_j = obs_genes[j]
        
        # Identify cell lines where BOTH genes have non-zero excess mutations
        co_occurrence_mask = (sub_matrix[:, i] > 0.0) & (sub_matrix[:, j] > 0.0)
        co_occurring_cells_count = np.sum(co_occurrence_mask)
        
        if co_occurring_cells_count < 2:
            continue # No edge if they never mutate together less than 2 leading edge cell lines
            
        # Extract the sum of the pair's scores for co-occurring cells
        pair_sum = sub_matrix[co_occurrence_mask, i] + sub_matrix[co_occurrence_mask, j]
        # Extract the corresponding cell pathway totals
        pathway_total_subset = cell_pathway_totals[co_occurrence_mask, 0]
        
        # Calculate the proportional weight per cell, then average them
        relative_dominance_per_cell = pair_sum / pathway_total_subset
        avg_edge_weight = np.mean(relative_dominance_per_cell)
        
        edge_accumulator.append({
            'Source': gene_i,
            'Target': gene_j,
            'Edge_Weight': avg_edge_weight,
            'Co_Occurrence_Count': int(co_occurring_cells_count),
            'Co_Occurrence_Prevalence': co_occurring_cells_count / n_cells
        })
        
    edges_df = pd.DataFrame(edge_accumulator)
    return nodes_df, edges_df

# %%
agg_edge_df_list = []
agg_node_df_list = []

for tmp_gene_set_name in out_path:
    print(tmp_gene_set_name)
    tmp_gene_set = gene_set_to_use_dict[tmp_gene_set_name]
    tmp_gene_set_leading_edge_list = tmp_res.query('Pathway_Name == @tmp_gene_set_name').Leading_Edge_Cell_Lines.iloc[0]
    node_df, edge_df = construct_leading_edge_network(tmp_gene_set,tmp_gene_set_leading_edge_list,tmp_drug_excess_mutation_count_tbl)
    agg_edge_df_list.append(edge_df.assign(Pathway_Name = tmp_gene_set_name))
    agg_node_df_list.append(node_df.assign(Pathway_Name = tmp_gene_set_name))



# %%
def noisy_or_merge(series: pd.Series) -> float:
    """
    Computes the Noisy-OR probability: 1 - Prod(1 - w_i)
    Assumes weights are bounded between 0 and 1.
    """
    # Clip to safety bounds [0, 1] to prevent floating-point anomalies
    weights = np.clip(series.values, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - weights)

def aggregate_pathway_networks_probabilistic_product(
    pathway_nodes_list: list[pd.DataFrame], 
    pathway_edges_list: list[pd.DataFrame], 
    pathway_names: list[str]                
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolidates multiple pathway networks using a dual Noisy-OR product 
    formulation for edge weights to balance prevalence and intensity.
    """
    # --- 1. Consolidate Nodes (Noisy-OR) ---
    all_nodes = []
    for df, path_name in zip(pathway_nodes_list, pathway_names):
        if df.empty:
            continue
        temp = df.copy()
        temp['Pathway_Name'] = path_name
        all_nodes.append(temp)
        
    global_nodes_raw = pd.concat(all_nodes, ignore_index=True)
    
    global_nodes = global_nodes_raw.groupby('Gene').agg(
        Consolidated_Intensity=('Weight_Intensity', noisy_or_merge),
        Consolidated_Prevalence=('Weight_Prevalence', noisy_or_merge),
        Pathway_Count=('Pathway_Name', 'nunique'),
        Pathways=('Pathway_Name', lambda x: list(x.unique()))
    ).reset_index()
    
    # --- 2. Consolidate Edges using Dual Noisy-OR Product ---
    all_edges = []
    for df, path_name in zip(pathway_edges_list, pathway_names):
        if df.empty:
            continue
        temp = df.copy()
        sorted_pairs = np.sort(temp[['Source', 'Target']].values, axis=1)
        temp['Source'] = sorted_pairs[:, 0]
        temp['Target'] = sorted_pairs[:, 1]
        temp['Pathway_Name'] = path_name
        all_edges.append(temp)
        
    global_edges_raw = pd.concat(all_edges, ignore_index=True)
    
    # Run independent Noisy-OR merges on raw Edge Weight and Co-Occurrence Prevalence
    global_edges = global_edges_raw.groupby(['Source', 'Target']).agg(
        Noisy_OR_Intensity=('Edge_Weight', noisy_or_merge),
        Noisy_OR_Prevalence=('Co_Occurrence_Prevalence', noisy_or_merge),
        Total_Co_Occurrence=('Co_Occurrence_Count', 'sum'),
        Crosstalk_Index=('Pathway_Name', 'nunique'),
        Pathways=('Pathway_Name', lambda x: list(x.unique()))
    ).reset_index()
    
    # Calculate the final probabilistic product weight
    global_edges['Consolidated_Edge_Weight'] = (
        global_edges['Noisy_OR_Intensity'] * global_edges['Noisy_OR_Prevalence']
    )
    
    # Sort network by this consolidated metric
    global_edges = global_edges.sort_values(by='Consolidated_Edge_Weight', ascending=False).reset_index(drop=True)
    
    return global_nodes, global_edges

# %%

obs_paths_list = pd.concat(agg_node_df_list).Pathway_Name.unique()
agg_node_df, agg_edge_df = aggregate_pathway_networks_probabilistic_product(agg_node_df_list,agg_edge_df_list,obs_paths_list)

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
centrality = nx.degree_centrality(agg_G)
node_sizes = [1 + (centrality[node] * 10000) for node in agg_G.nodes()]

# %%
# Plotting the result
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(agg_G, pos, node_size=node_sizes, node_color='#4D96FF')
nx.draw_networkx_edges(agg_G, pos, alpha=0.2, edge_color='black')
nx.draw_networkx_labels(agg_G, pos, font_size=10, font_color='grey')

plt.title("Spectral (Fiedler Vector) Hilbert Layout")
plt.axis('off')
plt.show()
# %%

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
import plotly.graph_objects as go

def create_interactive_network_explorer(
    pos: dict,                  # dict of {gene: (x, y)} from your Hilbert layout
    global_nodes: pd.DataFrame, # Columns: ['Gene', 'Consolidated_Intensity', 'Consolidated_Prevalence', ...]
    xi: np.ndarray,             # 2D grid X coordinates from your contour step
    yi: np.ndarray,             # 2D grid Y coordinates
    zi: np.ndarray,             # 2D grid Z (density) coordinates (with NaNs for masked areas)
    output_html_path: str = "network_explorer.html"
):
    """
    Generates a high-performance, zoomable, interactive HTML visualizer
    combining contour edge densities with dynamic gene labels.
    """
    # 1. Map node positions to our DataFrame for vectorized Plotly rendering
    nodes_df = global_nodes.copy()
    nodes_df['X'] = nodes_df['Gene'].map(lambda g: pos[g][0] if g in pos else np.nan)
    nodes_df['Y'] = nodes_df['Gene'].map(lambda g: pos[g][1] if g in pos else np.nan)
    nodes_df = nodes_df.dropna(subset=['X', 'Y'])
    
    # Create descriptive hover text for each gene node
    nodes_df['Hover_Text'] = nodes_df.apply(
        lambda row: (
            f"<b>Gene:</b> {row['Gene']}<br>"
            f"<b>Consolidated Intensity:</b> {row['Consolidated_Intensity']:.3f}<br>"
            f"<b>Consolidated Prevalence:</b> {row['Consolidated_Prevalence']:.3f}<br>"
            f"<b>Pathways Involved:</b> {len(row['Pathways'])}<br>"
#            f"<b>Primary Pathways:</b> {', '.join(row['Pathways'][:3])}"
        ), axis=1
    )
    
    # 2. Build the Plotly Figure
    fig = go.Figure()
    
    # --- Layer 1: Background Edge Contours ---
    # We use a custom colorscale (e.g., Viridis or Blues) with transparency (opacity)
    if xi is not None and yi is not None and zi is not None:
        fig.add_trace(
            go.Contour(
                x=xi[:, 0], # 1D array of X coordinates
                y=yi[0, :], # 1D array of Y coordinates
                z=zi.T,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Edge Density / Functional Cohort Weight", len=0.4, y=0.3),
                contours=dict(coloring='heatmap', showlabels=False),
                line=dict(width=0), # Hides harsh contour line boundaries
                opacity=0.9,
                hoverinfo='skip' # Do not intercept hover signals intended for nodes
            )
        )
        
    # --- Layer 2: Foreground Interactive Gene Nodes ---
    # Size nodes by prevalence, color them by overall mutational intensity
    node_weights =  nodes_df['Consolidated_Prevalence'].to_numpy()
    marker_sizes = 1 + (node_weights * 25)
    node_names = nodes_df['Gene'].to_list()
    fig.add_trace(
        go.Scatter(
            x=nodes_df['X'],
            y=nodes_df['Y'],
            mode='markers', # 'text' enables on-plot labels
            text=nodes_df['Gene'], # The actual gene names
            textposition="top center",
            hovertext=nodes_df['Hover_Text'],
            hoverinfo='text',
            marker=dict(
                size=marker_sizes,
                color=nodes_df['Consolidated_Intensity'],
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title="Mutational Intensity (Noisy-OR)", len=0.4, y=0.8),
                line=dict(width=1.5, color='white'), # Clean white border around nodes
            ),
            # This is the secret to dynamic labels:
            # We hide the labels until the user zooms in close, preventing a giant text block
            textfont=dict(
                size=10,
                color="black"
            )
        )
    )
    threshold_steps = np.linspace(0.0, float(np.max(node_weights)), 11)
    slider_steps = []
    for val in threshold_steps:
        # Mask out nodes below the threshold (set marker size to 0)
        # Nodes above threshold retain their calculated size
        filtered_sizes = np.where(node_weights >= val, marker_sizes, 0)
        # Also hide text labels for masked nodes
        filtered_text = np.where(node_weights >= val, node_names, "")
        step = dict(
            method="restyle",
            label=f"{val:.2f}",
            args=[
                {
                    "marker.size": [filtered_sizes],
                    "text": [filtered_text]
                },
                [1]  # Target TRACE INDEX 1 (The Gene Nodes trace)
            ]
        )
        slider_steps.append(step)
    # --- Attach Slider Layout to Plotly ---
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Min Node Weight Threshold: "},
            pad={"t": 50},
            steps=slider_steps
        )],
        title="Interactive Gene Network Overlaid on Mutational Landscape",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    
    # 3. Apply Polished Layout settings
    fig.update_layout(
        title="Consolidated Genomic Vulnerability Landscape",
        title_font=dict(size=18),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, 
            range=[0, xi.max() if xi is not None else 60]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[0, yi.max() if yi is not None else 60]
        ),
        plot_bgcolor='rgb(245, 245, 245)', # Soft off-white background
        width=1100,
        height=900,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="monospace"
        ),
        hovermode='closest',
        dragmode='pan' # Sets pan as default drag behavior for easy canvas exploration
    )
    
    # Save to a standalone HTML file containing all necessary JS/CSS
    fig.write_html(output_html_path, auto_open=True)
    print(f"Interactive dashboard successfully generated: {output_html_path}")

# %%

create_interactive_network_explorer(pos,agg_node_df,exi,eyi,ezi,output_html_path='./img/lapatinib_c2_network.html')

# %%


