import pandas as pd
from pathlib import Path
# %%
cellsaurus_id_tbl = pd.read_csv("./data/metadata/cell-line.tsv",sep="\t")
# %%
sanger_model_id_tbl = pd.read_csv("./data/metadata/depmap_Model.csv")

sanger_model_passport_tbl = pd.read_csv("./data/metadata/sanger_model_list_20260323.csv")

sanger_model_dataset_avail_tbl = pd.read_csv("./data/metadata/model_dataset_availability_20260323.csv")
# %%
sanger_model_passport_tbl.loc[:,['model_id', 'sample_id','synonyms', 'model_name','RRID']].merge(cellsaurus_id_tbl.loc[:,['id','sy','ac']],left_on='RRID',right_on='ac',how='right').loc[:,['model_id','id','ac']]
# %%
sanger_model_id_tbl.loc[:,['ModelID','CellLineName','RRID']].merge(cellsaurus_id_tbl.loc[:,['id','ac']],left_on='RRID',right_on='ac',how='right')
