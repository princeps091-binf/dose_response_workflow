import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
dose_data_file = "/home/vipink/Documents/dose_response_workflow/data/GDSC2_public_raw_data_27Oct23/GDSC2_public_raw_data_27Oct23.csv"

anno_tbl_file = "/home/vipink/Documents/dose_response_workflow/data/metadata/sanger_model_list_20260323.csv"

dose_fit_file = "/home/vipink/Documents/dose_response_workflow/data/GDSC2_fitted_dose_response_27Oct23.csv"
# %%
dose_coef_tbl = pd.read_csv(dose_fit_file,sep='\t')
# %%
dose_data_tbl = pd.read_csv(dose_data_file)
# %%
anno_tbl = pd.read_csv(anno_tbl_file)
# %%
dose_data_tbl.assign(single_dose = lambda df: df.TAG.str.contains("^L12-*")).query('single_dose')

# %%
dose_data_tbl.assign(combo_dose = lambda df: df.TAG.str.contains("UN")).query('combo_dose').shape
# %%
ax = dose_coef_tbl.plot.scatter(x='AUC',y='LN_IC50',alpha=0.02)
# %%
fig_new = plt.figure()
fig_new.set_label('New Window Plot')
# %%
# 3. Remove the axes from its original figure
ax.remove()
# %%
# 4. Add the axes to the new figure
# The 'add_axes' method works well here for positioning [left, bottom, width, height]
# [0, 0, 1, 1] means it will fill the entire new figure
fig_new.add_axes(ax) 
# %%
# 5. Display the new figure(s)
# This will block execution until all figures are closed in an interactive environment
plt.show() 
