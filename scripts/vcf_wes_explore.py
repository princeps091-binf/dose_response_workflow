import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
from src.utils.io import import_vcf_as_df, get_vcf_summary_tbl, extract_gene_anno, parse_gmt
from src.mutation.gene_set_analysis import *
# %%
vcf_folder = Path("/home/vipink/Documents/dose_response_workflow/data/omics/mutations_wes_vcf_20250226/")
vcf_file_list = list(vcf_folder.glob("*.gz"))
gene_set_file = "/home/vipink/Documents/dose_response_workflow/data/metadata/c2.all.v2026.1.Hs.symbols.gmt"

# %%
all_wes_mutation_df = pd.concat([get_vcf_summary_tbl(vcf_file) for vcf_file in vcf_file_list]).drop_duplicates()

gene_set_dict = parse_gmt(gene_set_file)
Gene_Set_size_tbl = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).explode('Genes').Gene_Set.value_counts().reset_index().rename(columns={'count':'gene_count'})

Gene_Set_avg_gene_in_count_df = pd.DataFrame.from_dict(list(gene_set_dict.items())).rename(columns={0:'Gene_Set',1:'Genes'}).explode('Genes').merge(all_wes_mutation_df.loc[:,['sanger_model_id','gene']],how='left',left_on='Genes',right_on='gene').assign(out = lambda df: ~df.gene.isna()).groupby(['Gene_Set','sanger_model_id']).agg(gene_in = ('out','sum')).reset_index()

# %%
tmp_cell_model = 'SIDM00146'
tmp_cell_model_count_tbl = get_cell_line_pathway_mutation_enrichment_tbl(Gene_Set_avg_gene_in_count_df,tmp_cell_model)

norm = mcolors.TwoSlopeNorm(vmin=tmp_cell_model_count_tbl.z_score.min(), vcenter=0, vmax=tmp_cell_model_count_tbl.z_score.max())
tmp_ax = tmp_cell_model_count_tbl.sort_values('z_score').plot.scatter(x='avg_gene',y='gene_in',c='z_score',cmap='coolwarm',s=10,logx=True,logy=True,norm=norm)
plt.show()

# %%
tmp_cell_model_count_tbl.query('z_score > 1').sort_values('z_score',ascending=False).head(20)

tmp_ax = tmp_cell_model_count_tbl.sort_values('z_score').assign(candidates = lambda df: np.where(df.z_score.gt(2),'red','grey')).plot.scatter(x='avg_gene',y='gene_in',logx=True,logy=True,c='candidates')
plt.show()


# %%
Gene_Set_tot_count_tbl = Gene_Set_avg_gene_in_count_df.groupby('Gene_Set').agg(tot_count = ('gene_in','sum')).reset_index()
cell_id_number = Gene_Set_avg_gene_in_count_df.sanger_model_id.nunique() 
total_zscore_tbl = parallel_zscore_estimation(Gene_Set_avg_gene_in_count_df,Gene_Set_tot_count_tbl,cell_id_number, n_cores=10)
# %%
tmp_cell_id = 'SIDM01567'
plt_tbl = total_zscore_tbl.query('cell_id == @tmp_cell_id')
cmap_norm =  mcolors.TwoSlopeNorm(vmin=plt_tbl.z_score.min(), vcenter=0, vmax=plt_tbl.z_score.max())
tmp_ax = plt_tbl.plot.scatter(x='predicted_count',y='gene_in',logx=False,logy=True,c='z_score',cmap='coolwarm',norm=cmap_norm)
plt.show()
# %%
def q1(x):
    return np.quantile(x,0.25)

def q3(x):
    return np.quantile(x,0.75)
stats = total_zscore_tbl.groupby('Gene_Set')['z_score'].agg(['median', q1, q3]).reset_index().assign(ypos = lambda df: df.loc[:,'median'].rank(method = 'first')).merge(total_zscore_tbl.Gene_Set.value_counts().reset_index().rename(columns={'count':'ncell'}))
# %%
import matplotlib.cm as cm
import matplotlib.colors as colors

fig, ax = plt.subplots(figsize=(10, 12))

norm = colors.Normalize(vmin=stats['ncell'].min(), 
                        vmax=stats['ncell'].max())
cmap = cm.get_cmap('viridis') # You can use 'magma', 'coolwarm', etc.
# %%
# Apply the colormap to each row
segment_colors = [cmap(norm(val)) for val in stats['ncell']]
# 2. Draw the IQR segments (Horizontal lines from Q1 to Q3)
ax.hlines(stats['ypos'], stats['q1'], stats['q3'], 
          color=segment_colors, linewidth=6, alpha=0.7, label='IQR (Q1-Q3)')

# 3. Draw the Median markers (Vertical dashes at the median)
# We use 'scatter' with a pipe symbol or 'vlines' for this
ax.vlines(stats['median'], [y - 0.4 for y in stats['ypos']], [y + 0.4 for y in stats['ypos']], 
          color='navy', linewidth=2, label='Median')

# 4. Add a vertical line at Z=0 for reference
ax.axvline(0, color='crimson', linestyle='--', alpha=0.5)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Number of Mutated Cell Lines')
# 5. Formatting the Y-axis with pathway names
ax.set_yticks(stats['ypos'])

ax.set_xlabel('Z-score (Mutation Enrichment)')
ax.set_title('Distribution of Pathway Mutation Injury')
ax.grid(axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
