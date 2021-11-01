#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
from keras.models import load_model
model=load_model('./Model-010.model')

labels_dict={0:'mask',1:' No mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

webcam = cv2.VideoCapture(0) 

# loading haar cascade classifier to detect face/faces
classifier = cv2.CascadeClassifier('C:/Users/admin/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
while True:
    (rval,im) = webcam.read() #to read each frame and to check if camera is on
    im=cv2.flip(im,1,1) #Flip to act as a mirror
    

    faces = classifier.detectMultiScale(im)# to detect a face/multiple faces 

    # Detect Face from the image
    for (x,y,w,h) in faces:
        face_img = im[y:y+h, x:x+w] #Save just the rectangle part of the face for further process
        resized=cv2.resize(face_img,(64,64))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,64,64,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        print(result)
        
        if(result*100>10):
            label=0
        else:
            label=1
        
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2) #highlight for face
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)#highlight for text
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)#to display text
     
    #cv2.namedWindow('Face Mask Detector', cv2.WINDOW_NORMAL)# just for a reference to change the size of the window
    #cv2.resizeWindow('Face Mask Detector',700,500) 
    #cv2.namedWindow('Face Mask Detector', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Face Mask Detector',im)# to Show the image
    key = cv2.waitKey(10) #delay for how long to keep the window open
   
    if key == ord('q'): 
        break # if q is pressed then break out of the loop 

webcam.release()# to Stop the video

cv2.destroyAllWindows()# to Close all started windows


# In[ ]:




