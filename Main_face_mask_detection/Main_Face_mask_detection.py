# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:57:36 2021

@author: HASHIB
"""

import cv2
import joblib

haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
model = joblib.load("mask_classifier_model.pkl")
names = {0 :'Mask', 1 : 'No Mask' }

while True:
     flag, img = capture.read()#-Read image from frame by frame
     if flag:
          faces = haar_data.detectMultiScale(img)#-detect faces in an frames
          for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 4)#-
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            face = face.reshape(1, -1)
            pred = model.predict(face)[0]
            n    = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244,250,250), 2)
            print(n) 
          cv2.imshow("result", img)
          if cv2.waitKey(2) == 27:
              break
capture.release()
cv2.destroyAllWindows() 


