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
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
import seaborn as sn
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import plot_confusion_matrix


users_data = pd.read_csv('users_data.csv')[:500]

# =============================================================================
# plot correlation matrix for first look at dataset
# =============================================================================
fig = plt.figure(dpi = 200, figsize=(20,20))
corrMatrix = users_data[users_data.columns[~users_data.columns.isin([ 'Email', 'first_name', 'gender'])]].corr()
#ticks = ['Messages sent', 'Messages recieved', 'Out-degree centrality', 'In-degree centrality']
sn.heatmap(corrMatrix, annot=True, fmt='.2f', 
           vmin=-1, vmax=1, 
           square = True, 
          # xticklabels = ticks, yticklabels = ticks
           )
plt.title('Correlation Matrix User Data')
#plt.xticks(rotation=-40)
plt.show() 

# =============================================================================
# classification with gender as y
# =============================================================================



X =np.array(users_data[users_data.columns[~users_data.columns.isin([ 'Email', 'gender', 'first_name','gender_num'])]], dtype = float)
y = np.array(users_data['gender_num'])
y = [0 if np.isnan(i) else i for i in y]
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(X)
X = imp.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#X_valid = scaler.transform(X_valid)

pca = PCA(n_components=2).fit(X_train)
X_train_rd = pca.transform(X_train) 

# =============================================================================
# plot pca as scatter
# =============================================================================
c_gend = ['red' if n == 1 else 'blue' if n == 0 else 'grey' for n in y_train]
plt.figure(figsize=(10,6)) 
plt.xlabel('Principal Component 1') 
plt.ylabel('Principal Compnent 2') 
plt.axis('equal') 
plt.scatter(X_train_rd[:, 0], X_train_rd[:, 1], c = c_gend)

plt.show()

# =============================================================================
# perform gridsearch to find best parameters
# =============================================================================

clf = RandomForestClassifier()

param_grid = { 
    'n_estimators': [10, 100, 1000],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 50, 100, None],
    #'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    #'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    #'ccp_alpha': np.arange(0, 5, 0.5)
}



CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, scoring=make_scorer(f1_score, average='weighted'))
CV_clf.fit(X_train, y_train)

CV_clf.best_params_

clf = RandomForestClassifier(bootstrap = True, max_features = None, class_weight='balanced', criterion = 'entropy', max_depth = 50, min_samples_leaf = 2, n_estimators = 1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
score= f1_score(y_test, y_pred, average='weighted')
print(score)

imp_features = clf.feature_importances_
imp_feat_ind = np.argsort(imp_features)[::-1]
imp_feat_sorted = np.sort(imp_features)[::-1]
plt.plot(imp_feat_sorted, '.')
plt.title('Feature importance (ordered), Random forest classifier')

corrmatrix = np.corrcoef(X_train.T, y_train)
corr_y = corrmatrix[-1][:-1]
corr_y_sort = np.sort(corr_y)[::-1]
corr_y_ind = np.argsort(corr_y)[::-1]


### plot feature space of two most important dimensions/features
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

#i = imp_feat_ind[0]
#j= imp_feat_ind[1]
i = corr_y_ind[0]
j = corr_y_ind[1]
n_i = users_data.columns[~users_data.columns.isin([ 'Email', 'first_name', 'gender'])][i]
n_j = users_data.columns[~users_data.columns.isin([ 'Email', 'first_name', 'gender'])][j]
plt.figure(figsize=(10,10)) 
plt.xlabel(n_i) 
plt.ylabel('height in organisational chart') 
plt.axis('equal') 
plt.title('') 
h = .1

x_min, x_max = X_train[:, i].min() - .5, X_train[:, i].max() + .5 
y_min, y_max = X_train[:, j].min() - .5, X_train[:, j].max() + .5 
xx, yy = np.meshgrid( np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

x = np.zeros((np.shape(xx.ravel())[0], np.shape(X_train)[1])) 
x[:, i] = xx.ravel() 
x[:, j] = yy.ravel()

#x = np.c_[xx.ravel(), yy.ravel()]

Z = clf.predict_proba(x)[:, 1]

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=cm, alpha=.8) 
plt.xlim((x_min, x_max)) 
plt.ylim((y_min, y_max))

plt.scatter(X_train[:, i], X_train[:, j], c=y_train, cmap=cm_bright, edgecolors='k')
#plt.scatter(X_valid[ind_correct, i], X_valid[ind_correct, j], c=y_valid[ind_correct], cmap=cm_bright, edgecolors='k', alpha=0.2)
#plt.scatter(X_valid[ind_wrong, i], X_valid[ind_wrong, j], c=y_valid[ind_wrong], cmap = cm_bright, edgecolors='k', alpha=1)

plt.show()


# =============================================================================
# Regression with burnoutrisk_score as y
# =============================================================================

burnoutrisk_score_features = ['n_long_days', 'overwork_ratio_1_ah', 'overwork_tot_1']

columns = {n:i for i, n in enumerate(users_data.columns[~users_data.columns.isin([ 'Email', 'first_name', 'gender'])])}

columns_y = [columns[i] for i in burnoutrisk_score_features]
X_r = np.concatenate((X, y))
y = np.sum(X[columns_y], axis = 0)

