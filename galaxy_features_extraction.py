#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
galaxy_features_extraction.py
Tintin Nguyen
Final Project
This script extracts geometric and photometric features from galaxy images
"""
from astroNN.datasets import load_galaxy10
from tensorflow.keras import utils
import numpy as np
import cv2 as cv

# load images and lebels from Galaxy Zoo
imgs, labels = load_galaxy10()


# convert the labels to 10 categories
labels = utils.to_categorical(labels, 10)

# convert image files and labels to numpy arrays
labels = labels.astype(np.float32)
images = imgs.astype(np.float32)
accuracy = np.array([])

# feature extraction
N = images.shape[0] # number of images
features = np.empty([N, 24], np.float32)
dim = images.shape[1]
npixel = dim ** 2

# will take nearly an hour to finish
for i in range(N):  
    
    # PHOTOMETRIC FEATURES
    maxg = np.max(images[i, :, :, 0])
    maxr = np.max(images[i, :, :, 1])
    maxi = np.max(images[i, :, :, 2])
    
    meang = np.mean(images[i, :, :, 0])
    meanr = np.mean(images[i, :, :, 1])
    meani = np.mean(images[i, :, :, 2])
    
    maxgri = np.max(np.array([maxg, maxr, maxi]))
    
    img = imgs[i]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale
    blur = cv.blur(gray, (20,20)) # blur image to reduce noise and smoothen edges
    ret, thresh = cv.threshold(blur, maxgri*0.035, maxgri*0.965, cv.THRESH_OTSU) # apply thresholds
    #find the contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    
    # number of "high intensities" pixies in each band
    features[i, 0] = np.count_nonzero(images[i, :, :, 0] / maxg > 0.025)
    features[i, 1] = np.count_nonzero(images[i, :, :, 1] / maxr > 0.025)
    features[i, 2] = np.count_nonzero(images[i, :, :, 2] / maxi > 0.025)
    
    # differences between fluxes of each bands
    features[i, 3] = np.abs(meanr - meani)
    features[i, 4] = np.abs(meang - meanr)
    features[i, 5] = np.abs(meang - meani)
    
    # center fluxes (bulge)
    centerg = np.mean(images[i, 33:38, 33:38, 0])
    centerr = np.mean(images[i, 33:38, 33:38, 1])
    centeri = np.mean(images[i, 33:38, 33:38, 2])
    
    features[i, 6] = centerg
    features[i, 7] = centerr
    features[i, 8] = centeri
    
    features[i, 9] = np.abs(centerg - centerr)
    features[i, 10] = np.abs(centerr - centeri)
    features[i, 11] = np.abs(centerg - centeri)
    
    # gini coefficient
    count = 1
    sum1i = 0
    sum2i = 0
    
    # transform 2d image of each band into a sorted 1d array
    sorti = np.sort(images[i, :, :, 2].flatten())
    
    for j in range(npixel):
        sum1i += (2*count - (npixel) - 1) * sorti[j]
        sum2i += sorti[j]            
        count += 1
    # compute gini coefficient    
    features[i, 12] = (sum1i) / ((npixel - 1) * sum2i)
    
    # Asymmetry Factor
    M = cv.getRotationMatrix2D(((69-1)/2.0,(69-1)/2.0),180,1) #180o rotation matrix 
    dsti = cv.warpAffine(images[i, 25:46, 25:46, 2], M,(69,69)) # 180o rotated image
    
    I0i = images[i, :, :, 2].flatten()
    I180i = dsti.flatten()
    

    totali = 0
    for j in range(69**2):
        if I0i[j] != 0:
           totali += np.abs(I0i[j]-I180i[j]) / np.abs(I0i[j])
           
    features[i, 13] = totali
    
    # GEOMETRIC FEATURES
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    rect = cv.minAreaRect(cnt)
    width, height = rect[1]
    
    if perimeter != 0:
        features[i, 14] = (area) / ((perimeter) ** 2) # form factor 
    
    if (width + height) != 0:
        features[i, 15] = perimeter / (2 * (width + height)) # convexity
        
    # bounding rectangle to / fill factor
    if (width * height) != 0:
        features[i, 16] = area / (width*height)
        
    # elliptical fit
    if np.shape(cnt)[0] > 4:
        ellipse = cv.fitEllipse(cnt)
        a, b = ellipse[1] # semiminor axis, semimajor axis
        if (b+a) != 0:
            features[i, 17] = (b-a) / (b+a) # eccentricity
    
    # M20 - light intensity distribution profile metric
    if np.shape(cnt)[0] > 4:
        xc = ellipse[0][0]
        yc = ellipse[0][1]
        
        Mtotg = 0 
        Mtotr = 0
        Mtoti = 0
        for x in range(dim):
            for y in range(dim):
                dist = ((x - xc) ** 2 + (y - yc) ** 2)
                fg = images[i, x, y, 0]
                Mtotg += fg * dist
                fr = images[i, x, y, 1]
                Mtotr += fr * dist
                fi = images[i, x, y, 2]
                Mtoti += fi * dist
        
        ftotg = np.sum(images[i, :, :, 0])
        fsortg = np.flip(np.sort(images[i, :, :, 0].flatten()))
        
        ftotr = np.sum(images[i, :, :, 1])
        fsortr = np.flip(np.sort(images[i, :, :, 1].flatten()))
        
        ftoti = np.sum(images[i, :, :, 2])
        fsorti = np.flip(np.sort(images[i, :, :, 2].flatten()))
        
        fsumg = 0
        fsumr = 0
        fsumi = 0
        count = 0
        while fsumg < (0.2*ftotg):
            fsumg += fsortg[count]
            count += 1
        while fsumr < (0.2*ftotr):
            fsumr += fsortr[count]
            count += 1
        while fsumi < (0.2*ftoti):
            fsumi += fsorti[count]
            count += 1
    
        M20g = np.log10(fsumg / Mtotg)
        M20r = np.log10(fsumr / Mtotr)
        M20i = np.log10(fsumi / Mtoti)
        
        features[i, 18] = M20g
        features[i, 19] = M20r
        features[i, 20] = M20i
    
    features[i, 21] = np.where(labels[i] == 1)[0][0]


np.savetxt('galaxy_features2.txt', features)
    

