import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import reload
from scipy.stats import beta
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import xlmhglite
import networkx as nx
import math
from hilbertcurve.hilbertcurve import HilbertCurve


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


