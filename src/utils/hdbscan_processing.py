import networkx as nx
import pandas as pd
import multiprocessing as mp

_G = None
_POINTS = None

def _init_worker(tree_nx, pos_idx):
    """This runs ONCE when each worker process is born."""
    global _G, _POINTS
    _G = tree_nx
    _POINTS = pos_idx

def get_mutation_descendants(v):
    # This runs in a separate process
    # nx.descendants returns a set, so the intersection is very efficient
    return nx.descendants(_G, v).intersection(_POINTS)

def produce_hdb_cluster_summary_tbl(tree_edge_df,tree_nx,chrom_pos_summary_tbl,root,n_cores=10):
    # 1. Prepare the static data
    # Converting to a set is CRITICAL: intersection with a set is much faster than a list
    point_indices = set(range(0, root))
    unique_parents = tree_edge_df.parent.drop_duplicates().tolist()

    # 2. Define the worker function
    # 3. Execute in parallel
    # Use the number of available CPU cores
    with mp.Pool(processes= n_cores,initializer= _init_worker,initargs=(tree_nx,point_indices)) as pool:
        cluster_pos_id_list = pool.map(get_mutation_descendants, unique_parents)
    # %%
    hdb_cluster_df = pd.DataFrame({'hdb_cluster':tree_edge_df.parent.drop_duplicates(),'pos_idx':cluster_pos_id_list})


    hdb_cluster_df = hdb_cluster_df.explode('pos_idx').assign(pos = chrom_pos_summary_tbl.POS.iloc[hdb_cluster_df.explode('pos_idx').pos_idx.to_numpy()].to_numpy()).groupby('hdb_cluster').agg(start = ('pos','min'),end = ('pos','max'),n_event = ('pos','nunique')).reset_index().merge(hdb_cluster_df)
    return hdb_cluster_df
