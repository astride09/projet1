# USAGE
# python detect_face_parts.py --image images/example_01.jpg 

import cv2
from cv2 import *
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import os
import glob
from os.path import basename, splitext





# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input imageS, resize it, and convert it to grayscale
data_path = 'data/train'
data_dir_list = os.listdir(data_path)
label_list_name=[]	
for data in data_dir_list:
	data_list = os.listdir('data/train/'+data)

	for img in data_list:
		image = cv2.imread(data_path + '/'+ data+'/'+img, 1)
		#image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# detect faces in the grayscale image
		rects = detector(gray, 1)

			# loop over the face detections
		for (i, rect) in enumerate(rects):
				# determine the facial landmarks for the face region, then
				# convert the landmark (x, y)-coordinates to a NumPy array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# loop over the face parts individually
				for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
					# clone the original image so we can draw on it, then
					# display the name of the face part on the image
					clone = gray.copy()
					cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
						0.7, (0, 0, 255), 2)

					# loop over the subset of facial landmarks, drawing the
					# specific face part
					for (x, y) in shape[i:j]:
						cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

					# extract the ROI of the face region as a separate image
					(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
					if x==0 or y==0 or w==0 or h==0:
						if os.path.exists(data_path + '/'+ data+'/'+img):
							os.remove(data_path + '/'+ data+'/'+img)
						if os.path.exists("landmarksCK+1/train4/" +data+'/' + name + "/" + img):
							os.remove("landmarksCK+1/train4/" +data+'/' + name + "/" + img)

					roi = gray[y:y + h, x:x + w]
					try:
						roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
					except ZeroDivisionError:
						print("Plusieurs")
					except cv2.error:
						if os.path.exists(data_path + '/'+ data+'/'+img):
							os.remove(data_path + '/'+ data+'/'+img)
						if os.path.exists("landmarksCK+1/train4/" +data+'/' + name + "/" + img):
							os.remove("landmarksCK+1/train4/" +data+'/' + name + "/" + img)
					except FileNotFoundError:
						print('non trouvé')
					
					# show the particular face part
					if not os.path.exists("landmarksCK+1/train4/" + name + "/" +data ):
							os.makedirs("landmarksCK+1/train4/"  + name + "/" +data )
							
					cv2.imwrite("landmarksCK+1/train4/"  + name + "/" +data + "/" + img, roi)

				