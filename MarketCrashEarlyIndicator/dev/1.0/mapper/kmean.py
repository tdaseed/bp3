# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:00:08 2019

@author: TanL
"""

import pandas as pd
f = "C:\\bp3\\MarketCrashEarlyIndicator\\dev\\1.0\\mapper\\input\\input_mapper_v1.csv"
df = pd.read_csv(f)
X = df.iloc[:,2:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X)
X.columns = df.columns[1:]

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,random_state=0).fit(X)
df_km = pd.DataFrame(km.labels_)
df_labels = pd.concat([df, df_km],axis=1)
df_labels.to_csv("C:\\bp3\\MarketCrashEarlyIndicator\\dev\\1.0\\mapper\\output_kmeans.csv", index=None)