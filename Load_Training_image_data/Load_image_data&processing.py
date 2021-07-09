# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:38:18 2021

@author: HASHIB
"""
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib



#Load image data
with_mask = np.load("Training_Data_With_mask.npy")
with_out_mask = np.load("Training_Data_WithOut_mask.npy")

#Find the shape
print(with_mask.shape)
print(with_out_mask.shape)

#To change To data shape have same sixe
#To REmove last two Rows 
with_mask = with_mask[:-2]
print(with_mask.shape)
print(with_out_mask.shape)

#Further Processing we have to change the 3D array to 2D array

with_mask = with_mask.reshape(652,50*50*3)
with_out_mask = with_out_mask.reshape(652,50*50*3)
print(with_mask.shape)
print(with_out_mask.shape)

#And Then concatenate the data into a single array:
#In Python np.r_ is used to add two array in ROW wise

X = np.r_[with_mask, with_out_mask]

print(X.shape)

# 'X' is the feature metrix
#Then we Need to Target metrix so create Zero metrix with
# Equal Row to the X
#create a (X.shape[0] = 1304 ) Row sized
labels = np.zeros(X.shape[0])

#In hear Using np.r_ in row based merging so
#652 first row is With Mask image another is without mask
half_sized_X = 1304/2
labels[int(half_sized_X):] = 1.0

#Add The labels name
names = {0 :'Mask', 1 : 'No Mask' }

#We Need to Applay the Mechine Learning Algorthm
#Hear we need to choose SVM Algorithm and its SVC class
#because its a label based Clasification Algorthim

#create a SVM model
svm = SVC(kernel="linear")

#split our model

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size= 0.25)

#Fit our training data

svm.fit(x_train, y_train)
#predicted the value in test set

y_pred = svm.predict(x_test)

#Accuracy score of predicted test
score=accuracy_score(y_test, y_test)

# The Accuracy score is

print("Model Accuracy is ", score)

#~~~~~~~~Saving trained model~~~~~~~~~~
joblib.dump(svm, "mask_classifier_model.pkl")