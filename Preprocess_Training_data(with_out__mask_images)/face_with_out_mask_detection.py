# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:52:02 2021

@author: HASHIB
"""
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Collect Face data with and Without Mask
#Read the mask image

path = glob.glob("C:/Users/HASHIB/Desktop/Face detections/FaceMaskdetection/Dataset/train/without_mask/*.jpg")
without_mask = []
haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for image in path:
     n = cv2.imread(image)
     without_mask.append(n)
length = len(without_mask)

data =[]
for img in without_mask:
       
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
          cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 4)

          face = img[y:y+h, x:x+w, :]
          face = cv2.resize(face, (50,50))
        if len(data) < length:
              data.append(face)
        #cv2.imshow("Result", img)       
        if cv2.waitKey(2) == 27 or len(data) >= length:
                  break
cv2.destroyAllWindows() 
 
#if len(data) == length:
 #np.save('Training_Data_WithOut_mask.npy',data)
plt.imshow(data[0])