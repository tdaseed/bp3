# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:07:43 2018

@author: tieqiangli
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep

#num_round = 100
#param['nthread'] = 4
#param['eval_metric'] = 'auc'

def logregobj_TL(preds, dtrain):
    labels = dtrain.get_label()
#    preds = 1.0 / (1.0 + np.exp(-preds))
    # biased punishment on the false positive predictions
    preds[preds>0.5] = 1.0 / (1.0 + np.exp(-preds[preds>0.5]))
    preds[preds<=0.5] = preds[preds<=0.5]

    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


'''
build backtesting batches
'''
f = 'data_price.csv'
df = pd.read_csv(f)

s = 4 
 
for i in range(1,13):
    df_tmp = df.drop(df.index[0:i*s])
    df_tmp = df_tmp.reset_index(drop=True)
    st = df_tmp.Date[0][-2:] + df_tmp.Date[0][3:5] + df_tmp.Date[0][:2]
    df_tmp.to_csv("data_price_20" + st + ".csv", index=False)
    


"""
x TDA
"""
param = {'bst:max_depth':5, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic', 'nthread':4,'eval_metric':'auc'}

num_round = 10

file_DF_tr = "DF_tr.csv"
file_DF_va = "DF_va.csv"
file_DF_te = "DF_te.csv"

DF_tr = pd.read_csv(file_DF_tr)
DF_va = pd.read_csv(file_DF_va)
DF_te = pd.read_csv(file_DF_te)

# record training dataset
y_tr = DF_tr.y 
X_tr = DF_tr.iloc[:,:-1]
X_tr = X_tr.drop(['ticker'],axis=1)
X_tr = X_tr.astype(float)
# record valication dataset
y_va = DF_va.y 
X_va = DF_va.iloc[:,:-1] 
X_va = X_va.drop(['ticker'],axis=1)
X_va = X_va.astype(float)
# record testing dataset
y_te = DF_te.y 
X_te = DF_te.iloc[:,:-1]
X_te = X_te.drop(['ticker'],axis=1)
X_te = X_te.astype(float)

dtrain = xgb.DMatrix(X_tr,label=y_tr)
dval = xgb.DMatrix(X_va,label=y_va)

evallist  = [(dval,'eval'), (dtrain,'train')]
etf_model = xgb.train(param,dtrain,num_round, evallist)

dpred = xgb.DMatrix(X_va)
y_pred_etf = etf_model.predict(dpred)


"""
w TDA
"""
param_TDA = {'bst:max_depth':20, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic', 'nthread':4,'eval_metric':'auc'}

num_round_TDA = 10

file_X_TDA_tr = "TDA/TDA_features_tr.csv"
file_X_TDA_va = "TDA/TDA_features_va.csv"
file_X_TDA_te = "TDA/TDA_features_te.csv"

X_TDA_tr = pd.read_csv(file_X_TDA_tr)
X_TDA_va = pd.read_csv(file_X_TDA_va)
X_TDA_te = pd.read_csv(file_X_TDA_te)

X_TDA_tr = pd.concat([X_tr,X_TDA_tr.iloc[:,2:]],axis=1)
X_TDA_te = pd.concat([X_te,X_TDA_te.iloc[:,2:]],axis=1)
X_TDA_va = pd.concat([X_va,X_TDA_va.iloc[:,2:]],axis=1)

dtrain_TDA = xgb.DMatrix(X_TDA_tr,label=y_tr)
dval_TDA = xgb.DMatrix(X_TDA_va,label=y_va)

evallist_TDA  = [(dval_TDA,'eval'), (dtrain_TDA,'train')]

etf_model_TDA = xgb.train(param_TDA,dtrain_TDA,num_round_TDA,evallist_TDA)

    

dpred_TDA = xgb.DMatrix(X_TDA_va)
y_pred_etf_TDA = etf_model_TDA.predict(dpred_TDA)
