#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
galaxy_analysis.py
Tintin Nguyen
Final Project
This script plots the feature importances and confusion matrices of the overall
classification into 10 groups
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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2 as cv

features = np.loadtxt('galaxy_features2.txt', dtype=np.float32())

train_idx, test_idx = train_test_split(np.arange(features.shape[0]), test_size=0.1)
train_features, train_labels, test_features, test_labels = features[train_idx, 0:21], features[train_idx, 21], features[test_idx, 0:21], features[test_idx, 21]

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

fig, ax = plt.subplots(figsize=(10,10))
cm = confusion_matrix(test_labels, predict_labels)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cmn, annot=True, fmt='.2f')
sns.set(font_scale=1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
