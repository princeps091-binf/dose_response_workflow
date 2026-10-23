# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%

dose_data_tbl = pd.read_csv('./data/DepMap/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv')

mutation_tbl = pd.read_csv('./data/DepMap/OmicsSomaticMutations.csv',sep='\t')
