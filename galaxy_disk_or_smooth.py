#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
galaxy_disk_or_smooth.py

"""
from astroNN.datasets import load_galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup, galaxy10_confusion
from tensorflow.keras import utils
import numpy as np
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import cv2 as cv

features = np.loadtxt('galaxy_features2.txt', dtype=np.float32())

N = np.shape(features)[0]

for i in range(N):
    gtype = features[i, 21]
    
    if gtype == 1 or gtype == 2 or gtype == 3:
        features[i, 22] = 1 # smooth galaxies
    else:
        features[i, 22] = 0 # disk galaxies

train_idx, test_idx = train_test_split(np.arange(features.shape[0]), test_size=0.1)
train_features, train_labels, test_features, test_labels = features[train_idx, 0:21], features[train_idx, 22], features[test_idx, 0:21], features[test_idx, 22]

dt = RandomForestClassifier(n_estimators = 200, max_depth = 40, max_features = "sqrt", 
                            min_samples_split = 4, min_samples_leaf= 3)

# train decision tree with train data
dt.fit(train_features, train_labels)
predict_labels = dt.predict(test_features)
print("Accuracy:", accuracy_score(test_labels, predict_labels))

fig, ax = plt.subplots(figsize=(15,10))
plt.bar(range(len(dt.feature_importances_)), dt.feature_importances_)
plt.title("Feature Importance")
plt.show()

test = np.array([], str)
predict = np.array([], str)

for i in range(np.shape(test_labels)[0]):
    
    if test_labels[i] == 1:
        test = np.append(test, "Smooth")
    else:
        test = np.append(test, "Disk")
        
    if predict_labels[i] == 1:
        predict = np.append(predict, "Smooth")
    else:
        predict = np.append(predict, "Disk")

cm = confusion_matrix(test_labels, predict_labels, normalize = "all")
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f')
sns.set(font_scale=3)
plt.ylabel('Actual')
plt.xlabel('Predicted')
ax.xaxis.set_ticklabels(['Disk', 'Smooth'])
ax.yaxis.set_ticklabels(['Disk', 'Smooth'])
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

for i in range(N):
    if (features[i, 22] == 0):
        var0_a = np.append(var0_a, features[i, 3])
        var1_a = np.append(var1_a, features[i, 8])
        var2_a = np.append(var2_a, features[i, 12])
        var3_a = np.append(var3_a, features[i, 17])
    else:
        var0_b = np.append(var0_b, features[i, 3])
        var1_b = np.append(var1_b, features[i, 8])
        var2_b = np.append(var2_b, features[i, 12])
        var3_b = np.append(var3_b, features[i, 17])
        
fig, ax = plt.subplots(figsize=(15,10))
ax.hist(var0_a, bins=20, range=(0,10), histtype='step', color='r', density = True, label = "Disk galaxies", linewidth=2)
ax.hist(var0_b, bins=20, range=(0,10), histtype='step', color='b', density = True, label = "Smooth galaxies", linewidth=2)
ax.set(title='Intensity difference r - i', ylabel='density')
plt.legend(loc = "upper left")

fig, ax = plt.subplots(figsize=(15,10))
ax.hist(var2_a, bins=20, range=(0.4,0.8), histtype='step', color='r', density = True, label = "Disk galaxies", linewidth=2)
ax.hist(var2_b, bins=20, range=(0.4,0.8), histtype='step', color='b', density = True, label = "Smooth galaxies", linewidth=2)
ax.set(title='Gini coefficient', ylabel='density')
plt.legend(loc = "upper left")

fig, ax = plt.subplots(figsize=(15,10))
ax.hist(var3_a, bins=20, range=(0,0.6), histtype='step', color='r', density = True, label = "Disk galaxies", linewidth=2)
ax.hist(var3_b, bins=20, range=(0,0.6), histtype='step', color='b', density = True, label = "Smooth galaxies", linewidth=2)
ax.set(title='Eccentricity', ylabel='density')
plt.legend()

plt.tight_layout()
plt.show()

