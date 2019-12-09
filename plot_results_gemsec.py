# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:45:52 2019

@author: siddh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

neg_samp = [2,4,6,8,10,12,14,16,18]
polit =    [0.859, 0.856, 0.854,0.848,0.848,0.846,0.839,0.838,0.849]
company =  [0.663,0.633,0.617,0.618,0.603,0.61,0.622,0.596,0.599]
artists =  [0.475,0.374,0.343,0.303,0.275,0.255, 0.271,0,0]


# data
df_polit = pd.DataFrame({'x': neg_samp, 'y': polit })
df_company =  pd.DataFrame({'x': neg_samp, 'y': polit })
# plot

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,6))
fig.subplots_adjust(hspace = 1, wspace=.001)
ax[0].plot( 'x', 'y', data=df_polit, linestyle='-', marker='o', color='r')
#plt.plot( 'x', 'y', data=df_polit, linestyle='-', marker='o', color='r')
#plt.xlabel('Negative Samples')
#plt.ylabel('Modularity')
#plt.title('Politicians Dataset')
#plt.show()
