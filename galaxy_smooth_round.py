#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
galaxy_smooth_round.py
Tintin Nguyen
Final Project
This script classifies smooth galaxies into 3 groups: completely round, in between, cigar-shaped
"""

import numpy as np
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

features = np.loadtxt('galaxy_features2.txt', dtype=np.float32())

N = np.shape(features)[0]

i = 0
while i < np.shape(features)[0]:
    gtype = features[i, 21]
    
    if gtype == 1 or gtype == 2 or gtype == 3:
        features[i, 22] = gtype - 1
        i += 1
    else:
        features = np.delete(features, i, axis = 0)
        

train_idx, test_idx = train_test_split(np.arange(features.shape[0]), test_size=0.1)
train_features, train_labels, test_features, test_labels = features[train_idx, 0:21], features[train_idx, 22], features[test_idx, 0:21], features[test_idx, 22]

dt = RandomForestClassifier(n_estimators = 200, max_depth = 40, max_features = "sqrt", 
                            min_samples_split = 8, min_samples_leaf= 3)

# train decision tree with train data
dt.fit(train_features, train_labels)
predict_labels = dt.predict(test_features)
print("Accuracy:", accuracy_score(test_labels, predict_labels))

fig, ax = plt.subplots(figsize=(15,10))
plt.bar(range(len(dt.feature_importances_)), dt.feature_importances_)
plt.title("Feature Importance")
plt.show()

cm = confusion_matrix(test_labels, predict_labels)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f')
sns.set(font_scale=3)
plt.ylabel('Actual')
plt.xlabel('Predicted')
ax.xaxis.set_ticklabels(['Completely Round', 'In Between', 'Cigar-Shaped'])
ax.yaxis.set_ticklabels(['Completely Round', 'In Between', 'Cigar-Shaped'])
plt.show(block=False)


# separate the data
var0_a = np.array([])
var1_a = np.array([])
var2_a = np.array([])
var3_a = np.array([])
var0_b = np.array([])
var1_b = np.array([])
var2_b = np.array([])
var3_b = np.array([])
var0_c = np.array([])
var1_c = np.array([])
var2_c = np.array([])
var3_c = np.array([])

for i in range(np.shape(features)[0]):
    if (features[i, 22] == 0):
        var0_a = np.append(var0_a, features[i, 14])
        var1_a = np.append(var1_a, features[i, 17])
    elif (features[i, 22] == 1):
        var0_b = np.append(var0_b, features[i, 14])
        var1_b = np.append(var1_b, features[i, 17])
    else:
        var0_c = np.append(var0_c, features[i, 14])
        var1_c = np.append(var1_c, features[i, 17])

fig, ax = plt.subplots(figsize=(15,10))
ax.hist(var0_a, bins=20, range = (0.045, 0.075), histtype='step', color='r', density = True, label = "Round", linewidth=2)
ax.hist(var0_b, bins=20, range = (0.045, 0.075),histtype='step', color='b', density = True, label = "In between", linewidth=2)
ax.hist(var0_c, bins=20, range = (0.045, 0.075),histtype='step', color='g', density = True, label = "Cigar-shaped", linewidth=2)
ax.set(title='Form factor', ylabel='density')
plt.legend(loc = "upper left")

fig, ax = plt.subplots(figsize=(15,10))
ax.hist(var1_a, bins=40, range=(0,0.6), histtype='step', color='r', density = True, label = "Round", linewidth=2)
ax.hist(var1_b, bins=40, range=(0,0.6), histtype='step', color='b', density = True, label = "In between", linewidth=2)
ax.hist(var1_c, bins=40, range=(0,0.6), histtype='step', color='g', density = True, label = "Cigar-shaped", linewidth=2)
ax.set(title='Eccentricity', ylabel='density')
plt.legend(loc = "upper right")

plt.tight_layout()
plt.show()

