# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:47:47 2017

@author: tan.li
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

"""
x TDA
"""
param = {'bst:max_depth':5, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic', 'nthread':4,'eval_metric':'auc'}

num_round = 6 # 600 #500 # 5 #13

#file_DF_tr = r"C:\@data\ETF\i=7_TDA_L=52x8B\DF_tr.csv"
#file_DF_te = r"C:\@data\ETF\i=7_TDA_L=52x8B\DF_te.csv"
#file_DF_va = r"C:\@data\ETF\i=7_TDA_L=52x8B\DF_va.csv"

file_DF_tr = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/DF_tr.csv"
file_DF_te = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/DF_te.csv"
file_DF_va = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/DF_va.csv"

DF_tr = pd.read_csv(file_DF_tr)
DF_te = pd.read_csv(file_DF_te)
DF_va = pd.read_csv(file_DF_va)

# record training dataset
y_tr = DF_tr.y 
X_tr = DF_tr.iloc[:,:-1]
X_tr = X_tr.drop(['ticker'],axis=1)
X_tr = X_tr.astype(float)
# record training dataset
y_te = DF_te.y 
X_te = DF_te.iloc[:,:-1]
X_te = X_te.drop(['ticker'],axis=1)
X_te = X_te.astype(float)
# record valication dataset
y_va = DF_va.y 
X_va = DF_va.iloc[:,:-1] 
X_va = X_va.drop(['ticker'],axis=1)
X_va = X_va.astype(float)

dtrain = xgb.DMatrix(X_tr,label=y_tr)
dtest = xgb.DMatrix(X_te,label=y_te)
evallist  = [(dtest,'eval'), (dtrain,'train')]

etf_model = xgb.train(param,dtrain,num_round, evallist)
# with customised objective function
etf_model = xgb.train(param,dtrain,num_round, evallist, logregobj_TL)

"""
w TDA
"""
param_TDA = {'bst:max_depth':20, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic', 'nthread':4,'eval_metric':'auc'}

num_round_TDA = 3 # 5 #300 #16 #7 #100 #16 # 

file_X_TDA_tr = r"C:\@data\ETF\i=8_TDA_L=52x8C\TDA\TDA_features_tr.csv"
file_X_TDA_te = r"C:\@data\ETF\i=8_TDA_L=52x8C\TDA\TDA_features_te.csv"
file_X_TDA_va = r"C:\@data\ETF\i=8_TDA_L=52x8C\TDA\TDA_features_va.csv"

file_X_TDA_tr = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/TDA/TDA_features_tr.csv"
file_X_TDA_te = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/TDA/TDA_features_te.csv"
file_X_TDA_va = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/TDA/TDA_features_va.csv"

X_TDA_tr = pd.read_csv(file_X_TDA_tr)
X_TDA_te = pd.read_csv(file_X_TDA_te)
X_TDA_va = pd.read_csv(file_X_TDA_va)

'''
norm normalisation, deprecate for now
'''
#scaler = prep.MinMaxScaler()
## training dataset
#scaler.fit(X_TDA_tr.iloc[:,2:])
#df_tmp = pd.DataFrame(scaler.transform(X_TDA_tr.iloc[:,2:]), columns=X_TDA_tr.columns[2:])
#X_TDA_tr = pd.concat([X_tr,df_tmp],axis=1)
## testing dataset
#scaler.fit(X_TDA_te.iloc[:,2:])
#df_tmp = pd.DataFrame(scaler.transform(X_TDA_te.iloc[:,2:]), columns=X_TDA_te.columns[2:])
#X_TDA_te = pd.concat([X_te,df_tmp],axis=1)
## validation dataset
#scaler.fit(X_TDA_va.iloc[:,2:])
#df_tmp = pd.DataFrame(scaler.transform(X_TDA_va.iloc[:,2:]), columns=X_TDA_va.columns[2:])
#X_TDA_va = pd.concat([X_va,df_tmp],axis=1)

X_TDA_tr = pd.concat([X_tr,X_TDA_tr.iloc[:,2:]],axis=1)
X_TDA_te = pd.concat([X_te,X_TDA_te.iloc[:,2:]],axis=1)
X_TDA_va = pd.concat([X_va,X_TDA_va.iloc[:,2:]],axis=1)

#X_TDA_tr = pd.concat([X_tr,X_TDA_tr.iloc[:,2]],axis=1)
#X_TDA_te = pd.concat([X_te,X_TDA_te.iloc[:,2]],axis=1)
#X_TDA_va = pd.concat([X_va,X_TDA_va.iloc[:,2]],axis=1)

dtrain_TDA = xgb.DMatrix(X_TDA_tr,label=y_tr)
dtest_TDA = xgb.DMatrix(X_TDA_te,label=y_te)

evallist_TDA  = [(dtest_TDA,'eval'), (dtrain_TDA,'train')]

etf_model_TDA = xgb.train(param_TDA,dtrain_TDA,num_round_TDA,evallist_TDA)
# with customised objective function
etf_model_TDA = xgb.train(param_TDA,dtrain_TDA,num_round_TDA,evallist_TDA,logregobj_TL)
"""
feature importance
"""
importances_etf = etf_model.get_fscore()
importance_frame_etf = pd.DataFrame({'Importance_etf': list(importances_etf.values()), 'Feature': list(importances_etf.keys())})
importance_frame_etf.sort_values(by = 'Importance_etf', inplace = True)
importance_frame_etf.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')

importances_etf_TDA = etf_model_TDA.get_fscore()
importance_frame_etf_TDA = pd.DataFrame({'Importance_etf_TDA': list(importances_etf_TDA.values()), 'Feature': list(importances_etf_TDA.keys())})
importance_frame_etf_TDA.sort_values(by = 'Importance_etf_TDA', inplace = True)
importance_frame_etf_TDA.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'blue')

#columns_xBeta.append('dim0_KK1')
#columns_xBeta.append('dim0_KK2')
#columns_xBeta.append('dim0_KK3')
#columns_xBeta.append('dim0_KK4')
#columns_xBeta.append('dim0_KK5')

dpred = xgb.DMatrix(X_va)
y_pred_etf = etf_model.predict(dpred)

dpred_TDA = xgb.DMatrix(X_TDA_va)
y_pred_etf_TDA = etf_model_TDA.predict(dpred_TDA)
#y_pred = etf_model_TDA.predict(dpred)

fpr_etf, tpr_etf, thresholds_etf = roc_curve(y_va, y_pred_etf, pos_label=1)
auc_etf = auc(fpr_etf, tpr_etf)

fpr_etf_TDA, tpr_etf_TDA, thresholds_etf_TDA = roc_curve(y_va, y_pred_etf_TDA, pos_label=1)
auc_etf_TDA = auc(fpr_etf_TDA, tpr_etf_TDA)

plt.plot(fpr_etf, tpr_etf, lw=1, alpha=0.3,
             label='ROC of model_etf (AUC = %0.4f)' % (auc_etf))
#plt.plot(fpr_etf_TDA, tpr_etf_TDA, lw=1, alpha=0.3,
#             label='ROC of model_etf_TDA (AUC = %0.4f)' % (auc_etf_TDA))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('etf Model vs. etf_TDA Model')
plt.legend(loc="lower right")
#plt.savefig(r'C:\Users\tan.li\Documents\Investment\Risk\creditRiskModel\TEST\auc.png')
plt.show()

'''
precision recall curve
'''
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_va, y_pred_etf)
p, r, th = precision_recall_curve(y_va, y_pred_etf)

average_precision = average_precision_score(y_va, y_pred_etf_TDA)
p_TDA, r_TDA, th_TDA = precision_recall_curve(y_va, y_pred_etf_TDA)

plt.step(r, p, color='b', alpha=0.2,
         where='post')
plt.fill_between(r, p, step='post', alpha=0.2,
                 color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

plt.plot(th,p[:-1],lw=1,alpha=0.3,label='precision')
plt.plot(th,r[:-1],lw=1,alpha=0.3,label='recall')
plt.xlabel('Threshold')
plt.ylabel('Precision / Recall')
plt.legend(loc="lower left")
plt.title('precision/recall profile - etf Model')

#plt.hist(p)
#plt.hist(r)
'''
result w TDA feature
'''
plt.step(r_TDA, p_TDA, color='b', alpha=0.2,
         where='post')
plt.fill_between(r_TDA, p_TDA, step='post', alpha=0.2,
                 color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

plt.plot(th_TDA,p_TDA[:-1],lw=1,alpha=0.3,label='precision')
plt.plot(th_TDA,r_TDA[:-1],lw=1,alpha=0.3,label='recall')
plt.xlabel('Threshold')
plt.ylabel('Precision / Recall')
plt.legend(loc="lower left")
plt.title('precision/recall profile - etf_TDA Model')

etf_model_results = pd.concat([pd.DataFrame(p[:-1]),pd.DataFrame(r[:-1]),pd.DataFrame(th)], axis=1)
#model_out = r"C:\@data\ETF\i=2_TDA=norms@batch\etf_model\etf_recision_recall.csv"
#model_out = r"C:\@data\ETF\i=3_y@4w\etf_model\etf_recision_recall.csv"
#precision_out = r"C:\@data\ETF\i=4_TDA_full_PL_rebase100\model\etf_recision_recall.csv"
#model_out = r"C:\@data\ETF\i=8_TDA_L=52x8C\model\etf_recision_recall.csv"
#model_out = r"C:\@data\ETF\i=6_TDA_L=52x8A\model\etf_recision_recall.csv"
#precision_TDA_out = r"C:\@data\ETF\i=4_TDA_full_PL_rebase100\model\etf_recision_recall_TDA.csv"
#model_TDA_out = r"C:\@data\ETF\i=8_TDA_L=52x8C\model\etf_recision_recall_TDA.csv"
#model_TDA_out = r"C:\@data\ETF\i=7_TDA_L=52x8B\model\etf_recision_recall_TDA.csv"
#model_TDA_out = r"C:\@data\ETF\i=6_TDA_L=52x8A\model\etf_recision_recall_TDA.csv"

precision_out = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/model/etf_recision_recall.csv"
precision_TDA_out = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/model/etf_recision_recall_TDA.csv"

etf_model_results.to_csv(precision_out,index=False)
etf_model_TDA_results = pd.concat([pd.DataFrame(p_TDA[:-1]),pd.DataFrame(r_TDA[:-1]),pd.DataFrame(th_TDA)], axis=1)
etf_model_TDA_results.to_csv(precision_TDA_out,index=False)
'''
save model
'''
import pickle
model_out = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/model/etf_model.pickle.dat"
pickle.dump(etf_model, open(model_out, "wb"))
etf_model = pickle.load(open(model_out, "rb"))
model_TDA_out = "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/model/etf_model_TDA.pickle.dat"
pickle.dump(etf_model_TDA, open(model_TDA_out, "wb"))
etf_model_TDA = pickle.load(open(model_TDA_out, "rb"))
