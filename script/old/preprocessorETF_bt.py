# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:24:24 2017

@author: tan.li
"""

import pandas as pd
import numpy as np
import re

#f1 = r"C:\@data\ETF\raw\data_price_etf.csv"
#f2 = r"C:\@data\ETF\raw\data_static_etf_com.csv"

f1 = "/Users/tieqiangli/@bp3/data/raw/data_price.csv"
f2 = "/Users/tieqiangli/@bp3/data/raw/data_static_com.csv"

f1 = "data_price_20180516.csv"
f2 = "data_static_com.csv"

df = pd.read_csv(f1) 
th = 26
col_drop = df.columns[df.count()<th]
# remove NA etf
df = df.drop(col_drop,axis=1)
# static descriptors of each etf
df_static = pd.read_csv(f2)

'''
1. feature engineering
'''
# meta parameters
w = 52 # 52 # batch window size
s = 4 # 13 # stride size
n = 8 #10 #8 # number of batches
L = w*n + s + 3 # total length of the historical data in unit of weeks, need additional 4w for y
#L = w*n + s # total length of the historical data in unit of weeks, need additional 4w for y
df = df.iloc[0:L+1,] # do an exact 10 year (52 x 10) cycle
#df = df.iloc[52:L+1+52,] # do an exact 10 year (52 x 10) cycle
#df = df.iloc[104:L+1+104,] # do an exact 10 year (52 x 10) cycle
df = df.reset_index()
df = df.drop('index',axis=1)

num_fea = 14
arr_ret = np.zeros(4)
TH_y = 0.04 # 0.02 # return cut-off for binary target/classification
array_tmp = np.zeros(num_fea)
df_columns = ['y','r_4w','r_8w','r_13w','r_26w','r_52w','h_l_4w','h_l_8w','h_l_13w','h_l_26w','h_l_52w','std_13w','std_26w','std_52w']
df_w = pd.DataFrame(df_columns,columns = ['feature_name'])
DF = pd.concat([pd.DataFrame(df_columns),pd.DataFrame(df_static.columns)]).transpose()
DF.columns = DF.iloc[0,:]
prop_tr, prop_va = 80, 20

#folder_TDA = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/TDA"
folder_TDA = "/Users/tieqiangli/@bp3/data/backtesting_1/TDA"

#L = 104 # 81 # 
#L = w*n + s + 3 # total length of the historical data in unit of weeks, need additional 4w for y
#def PrepETF(window, stride, num_batch, total_hist_length, path_price, path_static, folder_TDA, folder_out):
DF_tr = DF.drop(0) # training dataset placeholder
DF_te = DF.drop(0) # testing dataset placeholder
DF_va = DF.drop(0) # validation dataset placeholder
k = 1
for i in range(0,L-w-s,s): # loop with step size = s (3 months = 1 quarter)
#for i in range(0,L-w-s-3,s): # loop with step size = s (3 months = 1 quarter)
    bgn = L - i
    end = L - (w+i)
    col2drop = ['Date']
    for etf in df.columns[1:]:
        if int(pd.DataFrame(df[etf][end:bgn]).count()) < w:
            col2drop.append(etf)
    df_tmp = df[end:bgn]
    df_TDA = df_tmp.drop(col2drop,axis=1)
    '''
    inner merge the tickers from price dataset and the tickers from static dataset
    '''
    col_TDA = pd.DataFrame(df_TDA.columns, columns=['ticker'])
    col_static = pd.DataFrame(df_static.iloc[:,0], columns=['ticker'])
    col_GOOD = pd.merge(col_TDA,col_static,how='inner')
    col_GOOD_ = col_GOOD.ticker.tolist()
    df_TDA = df_TDA[col_GOOD_]
    '''
    rebase the time series at 100
    '''
    df_rel = df_TDA/df_TDA.shift(-1)
    df_rel.iloc[-1,:] = np.ones(df_rel.shape[1])    
    df_TDA = pd.DataFrame(100*df_rel[::-1].cumprod(),columns=df_rel.columns)
    df_TDA = df_TDA[::-1]
#    print(df_TDA.columns)
    prop = 100*i/(L-w-16)
#    print prop
    str_bgn = re.sub('/','', df.Date[bgn])
    str_end = re.sub('/','', df.Date[end])
    if prop < prop_tr:
        df_TDA.to_csv(folder_TDA + "/tr/" + str(k).zfill(3) + "_" + "data_prices_" + str_bgn + "_" + str_end + ".csv", index=False)
#    elif prop >= prop_tr and prop < (prop_tr + prop_te):
#        df_TDA.to_csv(folder_TDA + "/te/" + str(k).zfill(3) + "_" + "data_prices_" + str_bgn + "_" + str_end + ".csv", index=False)
    elif prop >= prop_tr:
        df_TDA.to_csv(folder_TDA + "/va/" + str(k).zfill(3) + "_" + "data_prices_" + str_bgn + "_" + str_end + ".csv", index=False)
    # saving the full window for back-testing    
#    df_TDA.to_csv(folder_TDA + "/te/" + str(k).zfill(3) + "_" + "data_prices_" + str_bgn + "_" + str_end + ".csv", index=False)
    k = k + 1
#    print i
#    print('progress = ' + "{:.0%}".format(prop))
    print('progress = ' + "{:.0f}%".format(prop))

''' one last dump for the testing dataset '''
i = i + 2*s 
# add 1 to make up 7 weeks forward to the  exactly the cutoff date
bgn = L - i + 1 
end = L - (w+i) + 1

df_tmp = df[end:bgn]
df_TDA = df_tmp.drop(col2drop,axis=1)
'''
inner merge the tickers from price dataset and the tickers from static dataset
'''
col_TDA = pd.DataFrame(df_TDA.columns, columns=['ticker'])
col_static = pd.DataFrame(df_static.iloc[:,0], columns=['ticker'])
col_GOOD = pd.merge(col_TDA,col_static,how='inner')
col_GOOD_ = col_GOOD.ticker.tolist()
df_TDA = df_TDA[col_GOOD_]
'''
rebase the time series at 100
'''
df_rel = df_TDA/df_TDA.shift(-1)
df_rel.iloc[-1,:] = np.ones(df_rel.shape[1])    
df_TDA = pd.DataFrame(100*df_rel[::-1].cumprod(),columns=df_rel.columns)
df_TDA = df_TDA[::-1]
#    print(df_TDA.columns)
#prop = 100*i/(L-w-16)
#    print prop
str_bgn = re.sub('/','', df.Date[bgn])
str_end = re.sub('/','', df.Date[end])
df_TDA.to_csv(folder_TDA + "/te/" + str(k).zfill(3) + "_" + "data_prices_" + str_bgn + "_" + str_end + ".csv", index=False)


for i in range(0,L-w-s,s): # loop with step size = 13 (3 months = 1 quarter)
#for i in range(0,L-w-s-3,s): # loop with step size = s (3 months = 1 quarter)
    bgn = L - i
    end = L - (w+i)
#    print(df.Date[bgn])
#    print(df.Date[end])    
#    for etf in ['ADRA','BHH']: #df.columns:
    df_w = pd.DataFrame(df_columns,columns = ['feature_name'])
    for etf in df.columns[1:]:
        if int(pd.DataFrame(df[etf][end:bgn]).count()) == w:
#            print(etf)
            # y - target            
            arr_ret[0] = df[etf][end-s]/df[etf][end]-1 # 3m return from X cut-off time
            arr_ret[1] = df[etf][end-s-1]/df[etf][end]-1 # 3m + 1w return from X cut-off time
            arr_ret[2] = df[etf][end-s-2]/df[etf][end]-1 # 3m + 2w return from X cut-off time
            arr_ret[3] = df[etf][end-s-3]/df[etf][end]-1 # 3m + 3w return from X cut-off time
            if max(arr_ret) < 0: y_ret = np.mean(arr_ret)   # if all returns in the preceeding 4w are negative
            else:  y_ret = np.mean(arr_ret[arr_ret > 0]) # if there exists at least one positive ret
            if y_ret >= TH_y: array_tmp[0] = 1
            else: array_tmp[0] = 0
            # feature: momentum
            array_tmp[1] = df[etf][end]/df[etf][end+4]-1 # r_4w
            array_tmp[2] = df[etf][end]/df[etf][end+8]-1 # r_8w 
            array_tmp[3] = df[etf][end]/df[etf][end+13]-1 # r_13w 
            array_tmp[4] = df[etf][end]/df[etf][end+26]-1 # r_26w 
            array_tmp[5] = df[etf][end]/df[etf][bgn]-1 # r_52w 
            # print(r_4w)
            # feature: volatility
            array_tmp[6] = max(df[etf][end:end+4])/min(df[etf][end:end+4]) # h_l_4w 
            array_tmp[7] = max(df[etf][end:end+8])/min(df[etf][end:end+8]) # h_l_8w 
            array_tmp[8] = max(df[etf][end:end+13])/min(df[etf][end:end+13]) # h_l_13w 
            array_tmp[9] = max(df[etf][end:end+26])/min(df[etf][end:end+26]) # h_l_26w     
            array_tmp[10] = max(df[etf][end:bgn])/min(df[etf][end:bgn]) # h_l_52w 
            #    std_4w
            #    std_8w 
            array_tmp[11] = np.std(np.divide(df[etf][end:end+13],df[etf][end+1:end+14])-1)*np.sqrt(52) # std_13w 
            array_tmp[12] = np.std(np.divide(df[etf][end:end+26],df[etf][end+1:end+27])-1)*np.sqrt(52) # std_26w 
            array_tmp[13] = np.std(np.divide(df[etf][end:bgn],df[etf][end+1:bgn+1])-1)*np.sqrt(52) # std_52w 
            df_w[etf] = pd.DataFrame(array_tmp)
            # transpose to merge with df_static
#    print(df_w.columns)
    df_w_ = pd.DataFrame(df_w.iloc[:,1:].transpose())
    df_w_.columns = df_w.iloc[:,0]
    df_w_['ticker'] = df_w_.index
#    df_w_.columns[0] = df_static.columns[0]
            
    # clean and merge with static features to form training batches
    df_W = pd.merge(df_w_, df_static, on = 'ticker', how='inner')
    df_W = df_W.sort_values(['ticker'])
    df_W = df_W.drop(['ticker'], axis=1)
    #array_df = np.append(array_df,np.transpose(array_tmp), axis = 0)
#    prop = i/(L-w)
#    prop = i/(L-w-16)
    prop = 100*i/(L-w-16)
    if prop < prop_tr:
        DF_tr = DF_tr.append(df_W,ignore_index=True) # keep appending for the training set when below prop_tr
#    elif prop >= prop_tr and prop < (prop_tr + prop_te):
#        DF_te = DF_te.append(df_W,ignore_index=True) # keep appending for the training set when below prop_te
    elif prop >= prop_tr:
        DF_va = DF_va.append(df_W,ignore_index=True) # keep appending for the training set when below prop_te
#    DF_te = DF_te.append(df_W,ignore_index=True) # keep appending for the testing set when below prop_te
#    print('progress = ' + "{:.0%}".format(prop))
    print('progress = ' + "{:.0f}%".format(prop))
#DF = DF.drop(0)
### feature: TDA


''' one last round of computation for the testing dataset '''
i = i + 2*s   
# add 1 to make up 7 weeks forward to the  exactly the cutoff date
bgn = L - i + 1
end = L - (w+i) + 1

#    print(df.Date[bgn])
#    print(df.Date[end])    
#    for etf in ['ADRA','BHH']: #df.columns:
df_w = pd.DataFrame(df_columns,columns = ['feature_name'])
for etf in df.columns[1:]:
    if int(pd.DataFrame(df[etf][end:bgn]).count()) == w:
#            print(etf)
        # y - target            
#        arr_ret[0] = df[etf][end-s]/df[etf][end]-1 # 3m return from X cut-off time
#        arr_ret[1] = df[etf][end-s-1]/df[etf][end]-1 # 3m + 1w return from X cut-off time
#        arr_ret[2] = df[etf][end-s-2]/df[etf][end]-1 # 3m + 2w return from X cut-off time
#        arr_ret[3] = df[etf][end-s-3]/df[etf][end]-1 # 3m + 3w return from X cut-off time
#        if max(arr_ret) < 0: y_ret = np.mean(arr_ret)   # if all returns in the preceeding 4w are negative
#        else:  y_ret = np.mean(arr_ret[arr_ret > 0]) # if there exists at least one positive ret
#        if y_ret >= TH_y: array_tmp[0] = 1
#        else: array_tmp[0] = 0
        array_tmp[0] = 0 # just dummy for now
        # feature: momentum
        array_tmp[1] = df[etf][end]/df[etf][end+4]-1 # r_4w
        array_tmp[2] = df[etf][end]/df[etf][end+8]-1 # r_8w 
        array_tmp[3] = df[etf][end]/df[etf][end+13]-1 # r_13w 
        array_tmp[4] = df[etf][end]/df[etf][end+26]-1 # r_26w 
        array_tmp[5] = df[etf][end]/df[etf][bgn]-1 # r_52w 
        # print(r_4w)
        # feature: volatility
        array_tmp[6] = max(df[etf][end:end+4])/min(df[etf][end:end+4]) # h_l_4w 
        array_tmp[7] = max(df[etf][end:end+8])/min(df[etf][end:end+8]) # h_l_8w 
        array_tmp[8] = max(df[etf][end:end+13])/min(df[etf][end:end+13]) # h_l_13w 
        array_tmp[9] = max(df[etf][end:end+26])/min(df[etf][end:end+26]) # h_l_26w     
        array_tmp[10] = max(df[etf][end:bgn])/min(df[etf][end:bgn]) # h_l_52w 
        #    std_4w
        #    std_8w 
        array_tmp[11] = np.std(np.divide(df[etf][end:end+13],df[etf][end+1:end+14])-1)*np.sqrt(52) # std_13w 
        array_tmp[12] = np.std(np.divide(df[etf][end:end+26],df[etf][end+1:end+27])-1)*np.sqrt(52) # std_26w 
        array_tmp[13] = np.std(np.divide(df[etf][end:bgn],df[etf][end+1:bgn+1])-1)*np.sqrt(52) # std_52w 
        df_w[etf] = pd.DataFrame(array_tmp)
        # transpose to merge with df_static
#    print(df_w.columns)
df_w_ = pd.DataFrame(df_w.iloc[:,1:].transpose())
df_w_.columns = df_w.iloc[:,0]
df_w_['ticker'] = df_w_.index
#    df_w_.columns[0] = df_static.columns[0]
        
# clean and merge with static features to form training batches
df_W = pd.merge(df_w_, df_static, on = 'ticker', how='inner')
df_W = df_W.sort_values(['ticker'])
#df_W = df_W.drop(['ticker'], axis=1)
#array_df = np.append(array_df,np.transpose(array_tmp), axis = 0)
#    prop = i/(L-w)
#    prop = i/(L-w-16)
#prop = 100*i/(L-w-16)
#if prop < prop_tr:
#    DF_tr = DF_tr.append(df_W,ignore_index=True) # keep appending for the training set when below prop_tr
##    elif prop >= prop_tr and prop < (prop_tr + prop_te):
##        DF_te = DF_te.append(df_W,ignore_index=True) # keep appending for the training set when below prop_te
#elif prop >= prop_tr:
#    DF_va = DF_va.append(df_W,ignore_index=True) # keep appending for the training set when below prop_te
DF_te = DF_te.append(df_W,ignore_index=True) # keep appending for the testing set when below prop_te
#    print('progress = ' + "{:.0%}".format(prop))
#print('progress = ' + "{:.0f}%".format(prop))

    
'''
2. preprocessing and archiving
'''
### set output paths
#f_out = "/Users/tieqiangli/@bp3/data/i=1_TDA"
#f_out = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x10"
#f_tr = f_out + "/DF_tr.csv"
#f_te = f_out + "/DF_te.csv"
#f_va = f_out + "/DF_va.csv"

f_tr = "DF_tr.csv"
f_te = "DF_te.csv"
f_va = "DF_va.csv"

### dump training/testing input as flat files for record
DF_tr.to_csv(f_tr,index=False,header=True)
DF_te.to_csv(f_te,index=False,header=True)
DF_va.to_csv(f_va,index=False,header=True)

# record training dataset
y_tr = DF_tr.y 
X_tr = DF_tr.iloc[:,:-1]
#X_tr = X_tr.drop(['ticker'],axis=1)
X_tr = X_tr.astype(float)
# record testing dataset
#y_te = DF_te.y 
#X_te = DF_te.iloc[:,:-1]
#X_te = X_te.drop(['ticker'],axis=1)
#X_te = X_te.astype(float)
# record valication dataset
y_va = DF_va.y 
X_va = DF_va.iloc[:,:-1] 
#X_va = X_va.drop(['ticker'],axis=1)
X_va = X_va.astype(float)

# record testing dataset
y_te = DF_te.y 
X_te = DF_te.iloc[:,:-1]
X_te = X_te.drop(['ticker'],axis=1)
X_te = X_te.astype(float)

'''
3. training input inspecting and statistics visualisation
'''

### training starts!

### result visulisation
