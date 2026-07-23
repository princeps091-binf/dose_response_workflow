import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import reload
from scipy.stats import beta, hypergeom
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import xlmhglite
import networkx as nx
import math
from hilbertcurve.hilbertcurve import HilbertCurve
import itertools
import plotly.graph_objects as go

import src.utils.io 
import src.mutation.gene_set_analysis
import src.dose_response.detect_response
import src.integration.gene_burden

def aggregate_gene_set_networks(graph_list):
    """
    Integrates a list of NetworkX graphs using Noisy-OR edge logic
    and tracks cross-set vertex prevalence.
    """
    Global_G = nx.Graph()
    M = len(graph_list)
    # Track vertex participation frequencies
    vertex_appearance_counts = {}
    vertex_total_burden = {}
    # 1. First pass: Collect node metadata across all layers
    for G in graph_list:
        for node, data in G.nodes(data=True):
            vertex_appearance_counts[node] = vertex_appearance_counts.get(node, 0) + 1
            # Assuming 'burden' holds your -log10(p-value) from your binomial step
            burden = data.get('burden', 0.0)
            vertex_total_burden[node] = vertex_total_burden.get(node, 0.0) + burden
    # 2. Add integrated nodes to the Global Graph
    for node in vertex_appearance_counts:
        prevalence_ratio = vertex_appearance_counts[node] / M
        avg_burden = vertex_total_burden[node] / vertex_appearance_counts[node]
        # Compute the Composite Support Index
        csi_score = prevalence_ratio * avg_burden
        Global_G.add_node(node, csi=csi_score, prevalence=prevalence_ratio)
    # 3. Second pass: Calculate Noisy-OR for Edges
    # Find every unique pair of nodes that share an edge in AT LEAST one graph
    all_possible_edges = set()
    for G in graph_list:
        all_possible_edges.update(G.edges())
    for u, v in all_possible_edges:
        # Collect weights for this specific edge across all graphs (0.0 if missing)
        layer_weights = []
        for G in graph_list:
            if G.has_edge(u, v):
                # Ensure your input weights are strictly bounded between [0, 1]
                w = G[u][v].get('weight', 0.0)
                layer_weights.append(w)
            else:
                layer_weights.append(0.0)
        # Apply Noisy-OR Equation: W = 1 - ∏(1 - w_i)
        complement_product = np.prod([1.0 - w for w in layer_weights])
        noisy_or_weight = 1.0 - complement_product
        # Only keep edges that clear a baseline noise threshold (e.g., > 0.05)
        if noisy_or_weight > 0:
            Global_G.add_edge(u, v, weight=noisy_or_weight)
    return Global_G
#-------------------

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

#------------------------------------------------

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

#-----------------------------------------------

def produce_cell_type_enrichment_tbl(drug_id,out_path,res_tbl,dose_tbl):
    leading_edge_cell_line_tbl = (res_tbl
                                  .query("Pathway_Name in @out_path")
                                  .loc[:,['Leading_Edge_Cell_Lines']]
                                  .explode('Leading_Edge_Cell_Lines')
                                  .drop_duplicates()
                                  )
    N_le = leading_edge_cell_line_tbl.shape[0]
    
    leading_count_tbl = (
            dose_tbl
           .query('DRUG_ID == @drug_id')
           .loc[:,['SANGER_MODEL_ID','CANCER_TYPE']]
           .query('SANGER_MODEL_ID in @leading_edge_cell_line_tbl.Leading_Edge_Cell_Lines.tolist()')
           .groupby('CANCER_TYPE')
           .agg(ncl = ('SANGER_MODEL_ID','nunique'))
           .reset_index()
                         )
    
    null_tbl = (dose_tbl
                .query('DRUG_ID == @drug_id')
                .loc[:,['SANGER_MODEL_ID','CANCER_TYPE']]
                .groupby('CANCER_TYPE')
                .agg(bg_n = ('SANGER_MODEL_ID','nunique'))
                .reset_index()
                )
    N_total = dose_tbl.query('DRUG_ID == @drug_id').loc[:,['SANGER_MODEL_ID']].drop_duplicates().shape[0]
    
    leading_enrich_stat_tbl = (
    leading_count_tbl.merge(null_tbl)
    .assign(
            pval = lambda df: hypergeom.sf(df.ncl -1,M=N_total,n=df.bg_n,N=N_le),
            expected_k = lambda df: hypergeom.stats(M=N_total,n=df.bg_n,N=N_le,moments='m'),
            variance_k = lambda df: hypergeom.stats(M=N_total,n=df.bg_n,N=N_le,moments='v'))
    .assign(z_score =lambda df: (df.ncl - df.expected_k) / np.sqrt(df.variance_k))
    )
    return leading_enrich_stat_tbl


#-----------------------------------------------
def spectral_hilbert_layout(G, gap_size=4):
    """
    Computes a Hilbert curve layout sorting nodes globally 
    via the Fiedler Vector (Spectral Sequencing).
    """
    ordered_nodes = []
    component_boundaries = []
    
    # 1. Sort components by size so large modules appear first
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    
    for comp in components:
        subgraph = G.subgraph(comp)
        
        if len(subgraph) > 2:
            # 2. Extract the Fiedler Vector for this component
            # algebraic_connectivity returns the eigenvalue, 
            # fiedler_vector returns the actual 1D layout vector
            try:
                fiedler = nx.fiedler_vector(subgraph, weight='w', method='scipy')
                
                # Pair nodes with their spectral coordinate and sort them
                node_fiedler_pairs = list(zip(subgraph.nodes(), fiedler))
                sorted_pairs = sorted(node_fiedler_pairs, key=lambda x: x[1])
                
                # Extract the sorted node list
                sorted_comp_nodes = [node for node, val in sorted_pairs]
            except Exception:
                # Fallback if spectral calculation diverges: sort by degree
                sorted_comp_nodes = sorted(subgraph.nodes(), key=lambda n: G.degree(n), reverse=True)
        else:
            sorted_comp_nodes = list(subgraph.nodes())
            
        ordered_nodes.extend(sorted_comp_nodes)
        component_boundaries.append(len(ordered_nodes))
        
    # 3. Calculate Hilbert Curve Capacity
    num_gaps = len(components) - 1
    total_estimated_slots = len(G.nodes) + (num_gaps * gap_size)
    
    p = math.ceil(math.log(total_estimated_slots, 4))
    total_capacity = 4**p
    
    # 4. Map to 1D Hilbert Distance with Component Gaps
    node_to_distance = {}
    current_distance = 0
    
    slack = total_capacity - total_estimated_slots
    step_slack = slack / max(1, len(G.nodes)) if slack > 0 else 0
    
    node_counter = 0
    for node in ordered_nodes:
        node_to_distance[node] = int(round(current_distance))
        node_counter += 1
        current_distance += 1 + step_slack
        
        # Inject structural gaps between disconnected biological components
        if node_counter in component_boundaries[:-1]:
            current_distance += gap_size
            
    # 5. Project onto 2D Hilbert Plane
    hc = HilbertCurve(p, n=2)
    pos = {}
    for node, dist in node_to_distance.items():
        dist = min(dist, total_capacity - 1)
        pos[node] = hc.point_from_distance(dist)
        
    return pos

def compute_distance_aware_edge_samples(pos, G_network, density_factor=3.0):
    """
    Samples points along edges while adjusting for Euclidean distance 
    to guarantee consistent topographic ridge density.
    """
    sample_x = []
    sample_y = []
    sample_z = []
    
    for u, v in G_network.edges():
        weight = G_network.get_edge_data(u, v)['weight']
        if weight <= 1e-10: 
            continue
            
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # 1. Calculate true Euclidean distance on the Hilbert canvas
        euclidean_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 2. Determine sample size dynamically based on distance AND weight
        # This guarantees long edges don't become "beaded necklaces"
        num_samples = int(max(5, np.ceil(euclidean_dist * density_factor * weight)))
        
        # 3. Linearly space the coordinates
        for t in np.linspace(0, 1, num_samples):
            sample_x.append(x1 + t * (x2 - x1))
            sample_y.append(y1 + t * (y2 - y1))
            sample_z.append(weight)
            
    return sample_x, sample_y, sample_z


def apply_topological_edge_mask(xi, yi, zi, sample_x, sample_y, threshold=0.6, actual_k =5):
    """
    Masks out meshgrid pixels that are too far from actual sampled edge traces.
    Prevents spatial bleedthrough across un-connected network territory.
    Memory-safe version using a k-d tree.
    """
    # 1. Bundle the scattered edge samples into an M x 2 matrix
    edge_points = np.column_stack([sample_x, sample_y])
    
    if len(edge_points) == 0:
        return zi
        
    # 2. Build a fast spatial lookup tree from the edge points
    # This takes virtually zero memory and builds in milliseconds
    tree = cKDTree(edge_points)
    
    # 3. Flatten the dense meshgrid coordinates
    grid_points = np.column_stack([xi.ravel(), yi.ravel()])
    
    # 4. Query the tree: Find the distance to the single NEAREST edge point
    # k=1 tells the tree to only look for the 1 nearest neighbor (no dense matrix allocated!)
    min_distances, _ = tree.query(grid_points, k=actual_k, workers=5) # workers=-1 uses all CPU cores
    # 5. Take the average distance of those k neighbors for each grid point
    # If k_neighbors was 1, we don't need to average (axis 1 won't exist)
    if actual_k > 1:
        avg_distances = np.mean(min_distances, axis=1)
    else:
        avg_distances = min_distances   
    # Reshape the 1D distance array back into the 2D matrix shape of our grid
    min_distances_grid = avg_distances.reshape(xi.shape)
    
    # 5. Apply the topological mask
    masked_zi = np.copy(zi)
    masked_zi[min_distances_grid > threshold] = np.nan
    
    return masked_zi

def generate_edge_contour_matrices(pos, G_network, grid_side, resolution=250):
    """
    Transforms raw network edges into dense uniform matrices 
    ready for Matplotlib contourf.
    """
    # 1. Extract the distance-aware scattered points
    sample_x, sample_y, sample_z = compute_distance_aware_edge_samples(
        pos, G_network, density_factor=5.0
    )
    
    # Guard against empty networks
    if len(sample_z) == 0:
        return None, None, None
        
    # 2. Build the target 2D coordinate grid (The uniform canvas)
    # This creates two 2D arrays matching the pixel resolution of your motherboard
    xi, yi = np.mgrid[0:grid_side:complex(resolution), 0:grid_side:complex(resolution)]
    
    # 3. Interpolate the scattered points onto the grid
    # 'linear' creates crisp, distinct ridge walls connecting your nodes
    zi = griddata(
        (sample_x, sample_y), sample_z, (xi, yi), 
        method='linear', fill_value=0.0
    )
    
    # 4. Optional: Apply an Alpha Mask (Fix from earlier)
    # This prevents the edges from bleeding into un-connected background territory
    zi = apply_topological_edge_mask(xi, yi, zi, sample_x, sample_y, threshold=0.7, actual_k = 5)
    
    return xi, yi, zi

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


