#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parameter_tuning.py
Tintin Nguyen
Final Project
This script does hyperparemeter tuning on the random forest classifier.
"""

import numpy as np
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


features = np.loadtxt('galaxy_features2.txt', dtype=np.float32())

# replace this line with the ones at the bottom to tune other parameters
# takes a bit of time
n_estimators = np.arange(1, 202, 5) 

accuracy = np.array([])
for i in n_estimators:
    train_idx, test_idx = train_test_split(np.arange(features.shape[0]), test_size=0.1)
    train_features, train_labels, test_features, test_labels = features[train_idx, 0:21], features[train_idx, 21], features[test_idx, 0:21], features[test_idx, 21]
    
    #n_estimators = np.arange(10, 201, 10)
    #min_samples_split = np.arange(2, 21)
    #max_depth = np.arange(1, 101, 2)
    #n_estimators = 200
    #max_features = sqrt
    #max_depth = 40
    dt = RandomForestClassifier(n_estimators = i, max_depth = 40, max_features = "sqrt", 
                                min_samples_split = 8, min_samples_leaf= 3)

    # train decision tree with train data
    dt.fit(train_features, train_labels)
    
    # apply decision tree model to test data
    predict_labels = dt.predict(test_features)

    accuracy = np.append(accuracy, accuracy_score(test_labels, predict_labels))

# for other parameters, edit this plot
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(n_estimators, accuracy, "blue")
ax.set_xlabel("Number of trees in random forest")
ax.set_ylabel("Accuracy")
ax.set_xticks(np.arange(0, 205, 20))
ax.plot(n_estimators, np.full(np.shape(n_estimators), 0.69), "red")
sns.set(font_scale=3)

"""
PARAMETERS TESTED:

# Number of trees in random forest
n_estimators = np.arange(1, 205, 5)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = np.arange(5, 105, 5)
# Minimum number of samples required to split a node
min_samples_split = np.arange(2, 22, 2)
# Minimum number of samples required at each leaf node
min_samples_leaf = np.arange(1, 21, 2)
# Method of selecting samples for training each tree
bootstrap = [True, False]
"""
    
        
        
    
    
    
    
    
    
    
    
    