# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:01:51 2019

@author: TanL
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

f1 = r"C:\mapper\input\high.csv"
f2 = r"C:\mapper\input\low.csv"

df_high = pd.read_csv(f1)
df_low = pd.read_csv(f2)

for fea in df_high.columns:
    t = ks_2samp(df_high[fea],df_low[fea])
    print(fea, t)
    

f_out = r"C:\mapper\output"

import seaborn as sns
import matplotlib.pyplot as plt

for fea in df_high.columns:
    print(fea)
    sns.distplot(df_low[fea],color='skyblue',label='low_price_property_cluster')
    sns.distplot(df_high[fea],color='red',label='high_price_property_cluster')
    plt.legend()
#    plt.show()
    fig = plt.figure()
    fig.savefig(f_out + "/" + fea + '.png', dpi=fig.dpi)
