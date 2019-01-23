# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:56:20 2018

@author: tanl
"""

'''
0. data preprocessing
'''

import pandas as pd
import numpy as np
#import xlwings as xw
#import xlsxwriter as xwr
#import re # for integer extraction

#f = r"N:\Risk_and_Performance\Public\Analytics\Risk@Weekly\WeeklyMarketAlert.xlsx"
#
#wb = xw.Book(f)
#ws = wb.sheets['input_BB']
#anchor = ws.range('anchor').value
#r_0 = int(re.findall('\d+', anchor)[0])
#r = ws.range('num_row').value
##r_M = int()
#c = ws.range('num_col').value
##c_N = xwr.utility.xl_col_to_name(int(1 + c)) # B -> a
#
#rgn = anchor + ':' + xwr.utility.xl_col_to_name(int(1 + c)) + str(int(r_0 + r))
#
#tbl = ws.range(rgn)
#df = pd.DataFrame(tbl.value)

f = r"N:\Risk_and_Performance\Public\Analytics\Risk@Weekly\input.csv"
df = pd.read_csv(f)
#df = df.dropna()
#cols = pd.DataFrame(df.columns)
# choose from one among the equity indices as the target
# EQY_DOW, EQY_SPX, EQY_NASDAQ, EQY_TSX, EQY_DAX, EQY_NIKKEI, EQY_TOPIX, EQY_HSI
tgt_ind = ['EQY_SPX']

### meta parameters
w = 26 # batch window size
s = 1 # the step size moving along to produce next batch/data point
d = 2 # the change window to calculate the drop
date_backtest = '26/12/2003'
TH = -0.05 # 72 positive incidences with 5% threshold; 13 with 10% threshold

'''
1. feature engineering
'''
# cols_sel = tgt_ind + ['BON_USD2Y','BON_USD5Y','BON_USD10Y','BON_USD30Y']
def fea_builder(df_t):
    fea = pd.DataFrame()
    fea['ret_1w_tgt'] = df_t.loc[0,tgt_ind]/df_t.loc[1,tgt_ind]-1
    fea['ret_2q_tgt'] = df_t.loc[0,tgt_ind]/df_t.loc[w-1,tgt_ind]-1
    fea['ret_1w_DXY'] = df_t.loc[0,'FX_DXY']/df_t.loc[1,'FX_DXY']-1
    fea['ret_2q_DXY'] = df_t.loc[0,'FX_DXY']/df_t.loc[w-1,'FX_DXY']-1
    fea['ret_1w_Gld'] = df_t.loc[0,'CTY_Gold']/df_t.loc[1,'CTY_Gold']-1
    fea['ret_2q_Gld'] = df_t.loc[0,'CTY_Gold']/df_t.loc[w-1,'CTY_Gold']-1
    fea['ret_1w_Oil'] = df_t.loc[0,'CTY_WTICrude']/df_t.loc[1,'CTY_WTICrude']-1
    fea['ret_2q_Oil'] = df_t.loc[0,'CTY_WTICrude']/df_t.loc[w-1,'CTY_WTICrude']-1
    
    fea['sprd_2y10y'] = df_t.loc[0,'BON_USD2Y'] - df_t.loc[0,'BON_USD10Y']
    fea['sprd_2y5y'] = df_t.loc[0,'BON_USD2Y'] - df_t.loc[0,'BON_USD5Y']

    fea['vol_tgt'] = np.std(np.divide(df_t.loc[0:w-2,tgt_ind],df_t.loc[1:w,tgt_ind])-1)*np.sqrt(52)
    fea['vol_DXY'] = np.std(np.divide(df_t.loc[0:w-2,'FX_DXY'],df_t.loc[1:w,'FX_DXY'])-1)*np.sqrt(52)
    fea['vol_Gld'] = np.std(np.divide(df_t.loc[0:w-2,'CTY_Gold'],df_t.loc[1:w,'CTY_Gold'])-1)*np.sqrt(52)
    fea['vol_Oil'] = np.std(np.divide(df_t.loc[0:w-2,'CTY_WTICrude'],df_t.loc[1:w,'CTY_WTICrude'])-1)*np.sqrt(52)

    return fea

'''
2. model training/validating
'''

'''
3. forecasting/backtesting
'''