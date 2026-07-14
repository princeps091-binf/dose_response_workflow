import pandas as pd
from pathlib import Path
import gzip
import fast_hdbscan
import numpy as np
import bioframe as bf
import networkx as nx
# %%

vcf_folder = Path("/home/vipink/Documents/dose_response_workflow/data/omics/mutations_wgs_vcf_20260305/")

vcf_file_list = list(vcf_folder.glob("*.gz"))
# %%

def import_vcf_as_df(vcf_file):
    with gzip.open(vcf_file, 'rt') as f:
        header_lines = sum(1 for line in f if line.startswith('##'))
    
    # Read into a DataFrame, skipping the header lines
    df = pd.read_csv(vcf_file, sep='\t', skiprows=header_lines)
    
    return df

# %%
all_wgs_mutations_df = pd.concat([import_vcf_as_df(f).loc[:,['#CHROM','POS']].assign(vcf= f.stem.split('_')[0]) for f in vcf_file_list]).rename(columns={'#CHROM':'chrom'})
# %%
all_wgs_mutations_vs_gene_df = bf.frac_gene_coverage(all_wgs_mutations_df.loc[:,['chrom','POS']].rename(columns={'POS':'start'}).assign(end = lambda df: df.start + 1),'hg38')
# %%
tmp_chrom = 'chr19'
chrom_wgs_mutation_df = all_wgs_mutations_df.query('chrom == @tmp_chrom')
# %%
chrom_pos_summary_tbl = chrom_wgs_mutation_df.value_counts().reset_index().groupby(['chrom','POS']).agg(avg_injury=('count','mean')).reset_index().merge(chrom_wgs_mutation_df.groupby(['chrom','POS']).agg(nsample=('vcf','nunique')).reset_index()).sort_values('nsample')
# %%
clusterer = fast_hdbscan.fast_hdbscan(chrom_pos_summary_tbl.loc[:,['POS']],sample_weights=chrom_pos_summary_tbl.nsample.to_numpy().astype(np.float32),min_cluster_size=2, return_trees= True)
chrom_pos_summary_tbl = chrom_pos_summary_tbl.assign(hdbscan_label = clusterer[0])
# %%
chrom_pos_summary_tbl.groupby('hdbscan_label').agg(start = ('POS','min'),end = ('POS','max'),n = ('POS','nunique')).reset_index().sort_values('n').assign(single = lambda df : df.n.lt(2)).query('~single').query('hdbscan_label >=0').assign(w = lambda df: df.end - df.start).w.quantile([0.05,0.25,0.5,0.75,0.95])


chrom_pos_summary_tbl.groupby('hdbscan_label').agg(start = ('POS','min'),end = ('POS','max'),n = ('POS','nunique')).reset_index().sort_values('n').assign(single = lambda df : df.n.lt(2)).query('~single').query('hdbscan_label >= 0').assign(w = lambda df: df.end - df.start).assign(mutation_rate = lambda df: df.n/df.w).w.mean()
# %%
chrom_cluster_coord_tbl = chrom_pos_summary_tbl.groupby('hdbscan_label').agg(chrom = ('chrom','first'),start = ('POS','min'),end = ('POS','max'),n = ('POS','nunique')).reset_index().query('hdbscan_label >= 0').loc[:,['chrom','start','end','n','hdbscan_label']]
# %%
chrom_cluster_coord_tbl = bf.frac_gene_coverage(chrom_cluster_coord_tbl,'hg38')
# %%
root = chrom_pos_summary_tbl.shape[0]

tree_edge_df = pd.DataFrame({'parent':clusterer[3].parent,'child':clusterer[3].child,'lambda':clusterer[3].lambda_val})

tree_nx = nx.from_pandas_edgelist(
    tree_edge_df, 
    source='parent', 
    target='child', 
    edge_attr=True,     # Keeps other columns as edge attributes
    create_using=nx.DiGraph()
)
# %%
import multiprocessing as mp

# 1. Prepare the static data
# Converting to a set is CRITICAL: intersection with a set is much faster than a list
point_indices = set(range(0, root))
unique_parents = tree_edge_df.parent.drop_duplicates().tolist()

# 2. Define the worker function
def get_descendants(v):
    # This runs in a separate process
    # nx.descendants returns a set, so the intersection is very efficient
    return nx.descendants(tree_nx, v).intersection(point_indices)

# 3. Execute in parallel
# Use the number of available CPU cores
with mp.Pool(processes=10 ) as pool:
    cluster_pos_id_list = pool.map(get_descendants, unique_parents)
# %%
hdb_cluster_df = pd.DataFrame({'hdb_cluster':tree_edge_df.parent.drop_duplicates(),'pos_idx':cluster_pos_id_list})


hdb_cluster_df = hdb_cluster_df.explode('pos_idx').assign(pos = chrom_pos_summary_tbl.POS.iloc[hdb_cluster_df.explode('pos_idx').pos_idx.to_numpy()].to_numpy()).groupby('hdb_cluster').agg(start = ('pos','min'),end = ('pos','max'),n_event = ('pos','nunique')).reset_index().merge(hdb_cluster_df)
# %%
hdb_cluster_df.explode('pos_idx').assign(pos = chrom_pos_summary_tbl.POS.iloc[hdb_cluster_df.explode('pos_idx').pos_idx.to_numpy()].to_numpy())

# %%
cluster_plot_df = tree_edge_df.query('child in @hdb_cluster_df.hdb_cluster.to_list()').groupby('child').agg(lambda_birth = ('lambda','min')).reset_index().rename(columns={'child':'hdb_cluster'}).merge(tree_edge_df.query('parent in @hdb_cluster_df.hdb_cluster.to_list()').groupby('parent').agg(lambda_death = ('lambda','max')).reset_index().rename(columns={'parent':'hdb_cluster'}).merge(hdb_cluster_df,how='right'),how='right').loc[:,['hdb_cluster','start','end','lambda_birth','lambda_death']]

cluster_plot_df = cluster_plot_df.replace([np.inf], 1, inplace=True)
# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_genomic_topography(rect_data,x_min,x_max, output_file="topography.png"):
    """
    rect_data: List of dicts with [x_start, x_end, y_bottom, y_top]
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    cluster_to_plot_list = bf.count_overlaps(cluster_plot_df.assign(chrom = 'chr2').loc[:,['chrom','start','end','hdb_cluster']],pd.DataFrame({'chrom':'chr2','start':[x_min],'end':[x_max]}).assign(start = lambda df: df.start.astype(int),end = lambda df: df.end.astype(int))).query('count > 0').hdb_cluster.unique()
    # Track max values for axis scaling
    max_y = 0
    to_plot_rect_data = rect_data.query('hdb_cluster in @cluster_to_plot_list')
    for rect in to_plot_rect_data.fillna(0).itertuples():
        width = rect.end - rect.start
        height = np.sqrt(rect.lambda_death) - np.sqrt(rect.lambda_birth)
        # Create a rectangle patch
        # Rectangle((x, y), width, height)
        polygon = patches.Rectangle(
            (rect.start, np.sqrt(rect.lambda_birth)), 
            width, 
            height, 
            linewidth=1, 
            edgecolor='blue', 
            facecolor='skyblue', 
            alpha=0.3  # Transparency reveals overlaps
        )
        
        ax.add_patch(polygon)
        
        # Update bounds for the plot
        max_y = max(max_y, rect.lambda_death)
    # Formatting the plot
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(0, np.sqrt(rect_data.lambda_death.max()) * 1.1)
    
    ax.set_xlabel('Genomic Position (bp)')
    ax.set_ylabel('Density Intensity')
    ax.set_title('Multi-scale Mutational Topography')
    
    # Use scientific notation for genomic coordinates if preferred
    ax.ticklabel_format(style='plain', axis='x') 
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Save the file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# %%
plot_genomic_topography(cluster_plot_df.fillna(0),1e6,1.5e6,'test.png')
# %%
bf.overlap(hdb_cluster_df.loc[:,['start','end','n_event']].assign(chrom = tmp_chrom,end = lambda df: df.end + 1).loc[:,['chrom','start','end','n_event']],chrom_wgs_mutation_df.rename(columns={'POS':'start'}).assign(end = lambda df: df.start + 1).loc[:,['chrom','start','end','vcf']])
