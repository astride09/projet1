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
import random

data_path1 = os.listdir('./landmarksCK+1/train4/jaw')

liste =[]
labels = []

for dataset in data_path1:

    print('load {}'.format(dataset))
    #print(img_list5)
    if dataset == 'neutral':
        img_list1=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+'neutral/' + img )
            cv2.imwrite('./New_CK+/test3/jaw/neutral/' + 'N_' + img, input_img )

    if dataset == 'angry':
        img_list1=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+'angry/' + img )
            cv2.imwrite('./New_CK+/test3/jaw/angry/' + 'A_' + img, input_img)

    if dataset == 'sad':
        img_list1=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+'sad/' + img )
            cv2.imwrite('./New_CK+/test3/jaw/sad/' + 'SA_' + img, input_img)

    if dataset == 'surprised':
        img_list1=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+'surprised/' + img )
            cv2.imwrite('./New_CK+/test3/jaw/surprised/' + 'Su_' + img, input_img)

    if dataset == 'happy':
        img_list1=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+'happy/' + img )
            cv2.imwrite('./New_CK+/test3/jaw/happy/' + 'H_' + img, input_img)

    if dataset == 'fearFful':
        img_list1=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+'fearful/' + img )
            cv2.imwrite('./New_CK+/test3/jaw/fearful/' + 'F_' + img, input_img)

    if dataset == 'disgusted':
        img_list1=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+'disgusted/' + img )
            cv2.imwrite('./New_CK+/test3/jaw/disgusted/' + 'D_' + img, input_img)



    

#data_path1 = os.listdir('./New_CK+/test')
test_data_path2 = os.listdir('./New_CK+/test3')

liste =[]
liste2=[]
liste3=[]
"""for dataset in data_path1:
        img_list1=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+''+ dataset+'/'+ img )
            liste.append(input_img)
            liste2.append(img)
            liste3.append('./landmarksCK+1/train4/jaw/'+'' + dataset +'/'+ img)"""

for dataset in test_data_path2:
        img_list1=os.listdir('./New_CK+/test3'+'/'+ dataset)
        img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
        for img in img_list1:
            input_img=cv2.imread('./New_CK+/test3/jaw/'+dataset +'/'+ img )
            liste.append(input_img)
            liste2.append(img)
            liste3.append('./New_CK+/test3/jaw/' +dataset+'/'+ img)


random.shuffle(liste3)
print(liste3)

labels = np.ones(len(liste3), dtype='int64')

r0 = (i for i,v in enumerate(liste3) if 'N_' in v)
r1 = (i for i,v in enumerate(liste3) if 'A_' in v)
r2 = (i for i,v in enumerate(liste3) if 'SA_' in v)
r3 = (i for i,v in enumerate(liste3) if 'Su_' in v)
r4 = (i for i,v in enumerate(liste3) if 'H_' in v)
r5 = (i for i,v in enumerate(liste3) if 'F_' in v)
r6 = (i for i,v in enumerate(liste3) if 'D_' in v)
print(r0)


import csv
"""title = ["label"]
with open('labal.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([title])"""
for k in r0:
    labels[k] = 0
for k in r1:
    labels[k] = 1
for k in r2:
    labels[k] = 2
for k in r3:
    labels[k] = 3
for k in r4:
    labels[k] = 4
for k in r5:
    labels[k] = 5
for k in r6:
    labels[k] = 6


with open('jaw_label.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([labels])
l=0
for im in liste3:
    im = cv2.imread(im)
    cv2.imwrite('base/test2/jaw/'+str(l)+'.jpg', im)
    l=l+1





"""setu = np.random.randint(7)
num = np.random.randint(10)
print(num)
print(setu)

#print(landmarksCK+1/train4_path5)
#print(landmarksCK+1/train4_path5)
j = 0
if setu == 0:
    dataset = 'neutral'
if setu == 1:
    dataset = 'Anger'
if setu == 2:
    dataset = 'Sadness'
if setu == 3:
    dataset = 'surprised'
if setu == 4:
    dataset = 'Happiness'
if setu == 5:
    dataset = 'fearful'
if setu == 6:
    dataset = '006'
lr = 'false'
print(len(os.listdir('./landmarksCK+1/train4/jaw/'+''+dataset)))
if len(os.listdir('./landmarksCK+1/train4/jaw/'+''+dataset)) != 0:
        lr = 'true'
        while lr == 'true':
            print('lol')   

            #for i in range(num):
            k=0
            img_list2=os.listdir('./landmarksCK+1/train4/jaw/'+ dataset)
            #print(img_list1)
            img_list1 = img_list2[:num]
            img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.png' or '.jpg' or '.jpeg']
            print(len(img_list1))
            print('load {}'.format(dataset))
            #print(img_list5)
            
            for img in img_list1:
                input_img=cv2.imread('./landmarksCK+1/train4/jaw/'+'' + dataset + '/' + img )
                liste.append(input_img)
                labels.append(setu)
                cv2.imwrite('base/test/'+str(j)+'.jpg', input_img)
                os.remove('./landmarksCK+1/train4/jaw/'+'' + dataset + '/' + img)

                j=j+1
            '''print(labels)
        print(len(liste))'''
else:
    setu = np.random.randint(7)
    num = np.random.randint(10)
    lr = 'true'
print(labels)
print(len(liste))
lr = 'false'"""
                                