#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import face_recogniton


# In[ ]:


import cv2


# In[ ]:


from imutils import paths


# In[ ]:


import os


# In[ ]:


import numpy as np


# In[ ]:


knownFace = ('C://Users/Gaby/Desktop/me1.jpg')


# In[ ]:


image = face.recogntion.load_image_file(knownFace)


# In[ ]:


face_locations = face_recogniton.face_locations(image, model="hog")


# In[ ]:


face_landmarks = face_recognition.face_landmarks(image)


# In[ ]:


(top,right,bottom,left)= face_locations[0]


# In[ ]:


desiredWidth = (right-left)


# In[ ]:


desiredHeight = (bottom-top)


# In[ ]:


align_f = alignFace(image, face_locations, face_landmarks, desiredWidth, desiredHeight)


# In[ ]:


known_face_encoding = face_recognition.face_encodings(align_f, num_jitters=10)[0]


# In[ ]:


unknownFace = ('C://GABYFLORES/Users/Gaby/Desktop/me2.jpg')


# In[ ]:


image = face.recogntion.load_image_file(unknownFace)


# In[ ]:


face_locations = face_recogniton.face_locations(image, model="hog")


# In[ ]:


face_landmarks = face_recognition.face_landmarks(image)


# In[ ]:


(top,right,bottom,left)= face_locations[0]


# In[ ]:


desiredWidth = (right-left)


# In[ ]:


desiredHeight = (bottom-top)


# In[ ]:


align_f = alignFace(image, face_locations, face_landmarks, desiredWidth, desiredHeight)


# In[ ]:


unknown_face_encoding = face_recognition.face_encodings(align_f, num_jitters=10)[0]


# In[ ]:


distance = face_recognition.face_distance([known_face_encoding], unkown_face_encoding)[0]


# In[ ]:


print("Distance : {}".format(distance))


# In[ ]:




