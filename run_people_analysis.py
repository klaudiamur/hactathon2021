#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:51:55 2021

@author: klaudiamur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import plot_confusion_matrix


user_data = pd.read_csv('users_data.csv')[:500]


X =np.array(users_data[users_data.columns[~users_data.columns.isin(['Location', 'Email', 'Name', 'gender', 'first_name', 'title', 'manager'])]], dtype = float)
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(X)
X = imp.transform(X)
scaler = StandardScaler().fit(X)
X_train = scaler.transform(X)
#X_valid = scaler.transform(X_valid)

pca = PCA(n_components=2).fit(X_train)
X_train_rd = pca.transform(X_train) 


kmeans = KMeans(n_clusters=16, random_state=0).fit_predict(X_train)
#### does it correspond to locations?


c_gend = ['red' if n == 1 else 'blue' if n == 0 else 'grey' for i, n in  users_data['gender_num'].iteritems()]
plt.figure(figsize=(10,6)) 
plt.xlabel('Principal Component 1') 
plt.ylabel('Principal Compnent 2') 
plt.axis('equal') 
plt.title('Wrong vs. correct classified validation datapoints') 

#plt.scatter(X_train_rd[:, 0], X_train_rd[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
plt.scatter(X_train_rd[:, 0], X_train_rd[:, 1], c = c_gend)
#plt.scatter(X_valid_rd[ind_correct, 0], X_valid_rd[ind_correct, 1], c=y_valid[ind_correct], cmap=cm_bright, edgecolors='k', alpha=0.2)
#plt.scatter(X_valid_rd[ind_wrong, 0], X_valid_rd[ind_wrong, 1], c=y_valid[ind_wrong], cmap=cm_bright, edgecolors='k', alpha=1)

plt.show()



users_data = data[:200]
X_train = users_data[['tot_msg_sent',
       'tot_msg_recieved', 'out_degree', 'in_degree', 'degree_centrality_G1',
       'betweenness_centrality_G1', 'closeness_centrality_G1',
       'eigenvector_centrality_G1', 'clustering_G1', 'core_number_G1',
       'node_clique_number_G1', 'constraint_G1', 'degree_centrality_G2',
       'betweenness_centrality_G2', 'closeness_centrality_G2',
       'eigenvector_centrality_G2', 'clustering_G2', 'core_number_G2',
       'node_clique_number_G2', 'constraint_G2', 'degree_centrality_G1_tot',
       'degree_centrality_G2_tot', 'overwork_ratio_1', 'overwork_ratio_1_ah',
       'overwork_tot_1', 'homophiliy_G_dir', 'homophiliy_G1', 'homophiliy_G2']][:170]
y_train = users_data['gender_num'][:170]
X_valid = users_data[['tot_msg_sent',
       'tot_msg_recieved', 'out_degree', 'in_degree', 'degree_centrality_G1',
       'betweenness_centrality_G1', 'closeness_centrality_G1',
       'eigenvector_centrality_G1', 'clustering_G1', 'core_number_G1',
       'node_clique_number_G1', 'constraint_G1', 'degree_centrality_G2',
       'betweenness_centrality_G2', 'closeness_centrality_G2',
       'eigenvector_centrality_G2', 'clustering_G2', 'core_number_G2',
       'node_clique_number_G2', 'constraint_G2', 'degree_centrality_G1_tot',
       'degree_centrality_G2_tot', 'overwork_ratio_1', 'overwork_ratio_1_ah',
       'overwork_tot_1', 'homophiliy_G_dir', 'homophiliy_G1', 'homophiliy_G2']][170:]
y_valid = users_data['gender_num'][170:]

### only use the ones that are in G2?

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

pca = PCA(n_components=2).fit(X_train)

X_valid_rd = pca.transform(X_valid)
X_train_rd = pca.transform(X_train)

fig = plt.figure(dpi = 200, figsize=(20,20))
corrMatrix = users_data[users_data.columns[~users_data.columns.isin(['Location', 'Email', 'Name'])]].corr()
#ticks = ['Messages sent', 'Messages recieved', 'Out-degree centrality', 'In-degree centrality']
sn.heatmap(corrMatrix, annot=True, fmt='.2f', 
           vmin=-1, vmax=1, 
           square = True, 
          # xticklabels = ticks, yticklabels = ticks
           )
plt.title('Correlation Matrix User Data')
#plt.xticks(rotation=-40)
plt.show() 


corrmatrix = users_data.corr()
corrmatrix = np.corrcoef(X_train.T, y_train)
corr_y = corrmatrix[-1][:-1]
corr_y_sort = np.sort(corr_y)

plt.figure(figsize = (10, 7))
plt.plot(corr_y_sort, '.')
plt.title('Correlation coefficients between features and y in training set')
plt.ylabel('Correlation coefficient')
plt.xlabel('Sorted features')