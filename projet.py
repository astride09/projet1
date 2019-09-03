from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Concatenate, Reshape, Input, concatenate, Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras import optimizers
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from keras.utils import np_utils
from keras.utils import to_categorical
import tensorflow as tf

import matplotlib.pyplot as plt
from keras.models import load_model
from keras import regularizers
import csv

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
a = ap.parse_args()
mode = a.mode 

label_train = np.ones(6160, dtype='int64')
with open('labaltrain.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
                

        r0 = (i for i,v in enumerate(reader) if '0' in v)
        r1 = (i for i,v in enumerate(reader) if '1' in v)
        r2 = (i for i,v in enumerate(reader) if '2' in v)
        r3 = (i for i,v in enumerate(reader) if '3' in v)
        r4 = (i for i,v in enumerate(reader) if '4' in v)
        r5 = (i for i,v in enumerate(reader) if '5' in v)
        r6 = (i for i,v in enumerate(reader) if '6_' in v)
        print(r0)


        import csv
        """title = ["label"]
        with open('labal.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([title])"""
        for k in r0:
                label_train[k] = 0
        for k in r1:
                label_train[k] = 1
        for k in r2:
                label_train[k] = 2
        for k in r3:
                label_train[k] = 3
        for k in r4:
                label_train[k] = 4
        for k in r5:
                label_train[k] = 5
        for k in r6:
                label_train[k] = 6

ytrain = label_train
ytrain = to_categorical(ytrain, 7)



label_test = np.ones(1540, dtype='int64')
with open('labaltest.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
                

        r0 = (i for i,v in enumerate(reader) if '0' in v)
        r1 = (i for i,v in enumerate(reader) if '1' in v)
        r2 = (i for i,v in enumerate(reader) if '2' in v)
        r3 = (i for i,v in enumerate(reader) if '3' in v)
        r4 = (i for i,v in enumerate(reader) if '4' in v)
        r5 = (i for i,v in enumerate(reader) if '5' in v)
        r6 = (i for i,v in enumerate(reader) if '6_' in v)
        print(r0)


        import csv
        """title = ["label_test"]
        with open('labal.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([title])"""
        for k in r0:
                label_test[k] = 0
        for k in r1:
                label_test[k] = 1
        for k in r2:
                label_test[k] = 2
        for k in r3:
                label_test[k] = 3
        for k in r4:
                label_test[k] = 4
        for k in r5:
                label_test[k] = 5
        for k in r6:
                label_test[k] = 6

ytest = label_test
ytest = to_categorical(ytest, 7)



'''labels = np.ones((7700),dtype='int64')
labels[0:1099]=0 #2848
labels[1100:2199]=1 #679
labels[2200:3299]=2 #940
labels[3300:4399]=3 #960
labels[4400:5499]=4 #1372
labels[5500:6599]=5 #715
labels[5600:7699]=6 #586 '''



'''

labels = np.ones((6160),dtype='int64')
labels[0:879]=0 #2848
labels[880:1759]=1 #679
labels[1760:2639]=2 #940
labels[2640:3519]=3 #960
labels[3520:4399]=4 #1372
labels[4400:5279]=5 #715
labels[5280:6159]=6 #586

names = ['NEUTRAL','ANGRY','SAD','SURPRISE','HAPPY','FEAR','DISGUST','CONTEMPT']

ytrain = labels
ytrain = to_categorical(ytrain, 7)

test_labels = np.ones((1540),dtype='int64')
test_labels[0:219]=0 #712
test_labels[220:439]=1 #170
test_labels[440:659]=2 #235
test_labels[660:879]=3 #240
test_labels[880:1099]=4 #344
test_labels[1100:1319]=5 #179
test_labels[1320:1539]=6 #146

ytest = test_labels
ytest = to_categorical(ytest, 7)
names = ['NEUTRAL','ANGRY','SAD','SURPRISE','HAPPY','FEAR','DISGUST','CONTEMPT']'''


#load dataset
train_data_path1 = os.listdir('./landmarksCK+1/train/right_eye')
test_data_path1 = os.listdir('./landmarksCK+1/test/right_eye')

train_data_path2 = os.listdir('./landmarksCK+1/train/left_eye')
test_data_path2 = os.listdir('./landmarksCK+1/test/left_eye')

train_data_path3 = os.listdir('./landmarksCK+1/train/right_eyebrow')
test_data_path3 = os.listdir('./landmarksCK+1/test/right_eyebrow')

train_data_path4 = os.listdir('./landmarksCK+1/train/left_eyebrow')
test_data_path4 = os.listdir('./landmarksCK+1/test/left_eyebrow')

train_data_path5 = os.listdir('./landmarksCK+1/train/mouth')
test_data_path5 = os.listdir('./landmarksCK+1/test/mouth')

train_data_path6 = os.listdir('./landmarksCK+1/train/nose')
test_data_path6 = os.listdir('./landmarksCK+1/test/nose')

train_data_path7 = os.listdir('./landmarksCK+1/train/jaw')
test_data_path7 = os.listdir('./landmarksCK+1/test/jaw')



img_data_list, img_data_list1, img_data_list2, img_data_list3, img_data_list4, img_data_list5, img_data_list6, img_data_list7, img_data_list8 = [], [], [], [], [], [], [], [], []

'''def load_jaffe():
        data_path = '/content/drive/My Drive/Colab Notebooks/jaffe'
        data_dir_list = os.listdir(data_path)
        for dataset in data_dir_list:
            img_list=os.listdir(data_path+'/'+ dataset)
            print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
            for img in img_list:
                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
                input_img_resize=cv2.resize(input_img,(256,256))
                img_data_list.append(input_img_resize)
                
        img_data = np.asarray(img_data_list)
        img_data = img_data.astype('float32')
        img_data = np.expand_dims(img_data, -1)
        img_data.shape

        num_of_samples = img_data.shape[0]
        labels = np.ones((num_of_samples,),dtype='int64')

        labels[0:29]=0 #30
        labels[30:58]=1 #29
        labels[59:90]=2 #32
        labels[91:121]=3 #31
        labels[122:151]=4 #30
        labels[152:182]=5 #31
        labels[183:]=6 #30

        names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
        return img_data, labels

faces, labels = load_jaffe()
        # convert class labels to on-hot encoding# conve 
labels = np_utils.to_categorical(labels, 7)
X_train, xtest,ytrain,ytest = train_test_split(faces, labels, test_size=0.2,shuffle=True)

        # Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5'''

#load right_eye dataset for train and test

print('load right_eye')
#print(img_list5)
for img in train_data_path1:
        input_img1=cv2.imread('./landmarksCK+1/train/right_eye' + '/'+ img )
        input_img1=cv2.resize(input_img1, (70,70))
        input_img1=cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
        img_data_list1.append(input_img1)

img_data1 = np.asarray(img_data_list1)
img_data1 = img_data1.astype('float32')
img_data1 = np.expand_dims(img_data1, -1)
#print(img_data5.shape)
#xtrain1, xtest1,ytrain1,ytest1 = train_test_split(img_data1, labels,test_size=0.2,shuffle=True)
'''xtrain1 = xtrain1.astype('float32')
xtest1 = xtest1.astype('float32')

xtrain1 /= 255
xtest1 /= 255

ytrain1 = to_categorical(ytrain1, 7)
ytest1 = to_categorical(ytest1, 7) '''

xtrain1 = img_data1
xtrain1 = xtrain1.astype('float32')
xtrain1 /= 255

img_data_list11 = []

for img in test_data_path1:
        input_img1=cv2.imread('./landmarksCK+1/test/right_eye' + '/'+ img )
        input_img1=cv2.resize(input_img1, (70,70))
        input_img1=cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
        img_data_list11.append(input_img1)

img_data1 = np.asarray(img_data_list11)
img_data1 = img_data1.astype('float32')
img_data1 = np.expand_dims(img_data1, -1)

xtest1 = img_data1
xtest1 = xtest1.astype('float32')
xtest1 /= 255 


print('load left_eye')
#print(img_list5)
for img in train_data_path2:
        input_img2=cv2.imread('./landmarksCK+1/train/left_eye' +'/'+ img )
        input_img2=cv2.resize(input_img2, (70,70))
        input_img2=cv2.cvtColor(input_img2, cv2.COLOR_BGR2GRAY)
        img_data_list2.append(input_img2)

img_data2 = np.asarray(img_data_list2)
img_data2 = img_data2.astype('float32')
img_data2 = np.expand_dims(img_data2, -1)


xtrain2 = img_data2
xtrain2 = xtrain2.astype('float32')
xtrain2 /= 255

img_data_list22 = []

    #print(img_list5)
for img in test_data_path2:
        input_img2=cv2.imread('./landmarksCK+1/test/left_eye' + '/'+ img )
        input_img2=cv2.resize(input_img2, (70,70))
        input_img2=cv2.cvtColor(input_img2, cv2.COLOR_BGR2GRAY)
        img_data_list22.append(input_img2)

img_data2 = np.asarray(img_data_list22)
img_data2 = img_data2.astype('float32')
img_data2 = np.expand_dims(img_data2, -1)

xtest2 = img_data2
xtest2 = xtest2.astype('float32')
xtest2 /= 255 

print('load right_eyebrow')
#print(img_list5)
for img in train_data_path3:
        input_img3=cv2.imread('./landmarksCK+1/train/right_eyebrow' + '/'+ img )
        input_img3=cv2.resize(input_img3, (70,70))
        input_img3=cv2.cvtColor(input_img3, cv2.COLOR_BGR2GRAY)
        img_data_list3.append(input_img3)

img_data3 = np.asarray(img_data_list3)
img_data3 = img_data3.astype('float32')
img_data3 = np.expand_dims(img_data3, -1)
'''
xtrain3, xtest3,ytrain3,ytest3 = train_test_split(img_data3, labels,test_size=0.2,shuffle=True)
xtrain3 = xtrain3.astype('float32')
xtest3 = xtest3.astype('float32')

xtrain3 /= 255
xtest3 /= 255
'''


xtrain3 = img_data3
xtrain3 = xtrain3.astype('float32')
xtrain3 /= 255

img_data_list33 = []

img_list3 = [img_list for img_list in test_data_path3 if img_list[-4:]=='.png' or '.jpg']
#print(img_list5)
for img in img_list3:
        input_img3=cv2.imread('./landmarksCK+1/test/right_eyebrow' + '/'+ img )
        input_img3=cv2.resize(input_img3, (70,70))
        input_img3=cv2.cvtColor(input_img3, cv2.COLOR_BGR2GRAY)
        img_data_list33.append(input_img3)

img_data3 = np.asarray(img_data_list33)
img_data3 = img_data3.astype('float32')
img_data3 = np.expand_dims(img_data3, -1)

xtest3 = img_data3
xtest3 = xtest3.astype('float32')
xtest3 /= 255 

#left_eyebrow dataset

img_list4 = [img_list for img_list in train_data_path4 if img_list[-4:]=='.png' or '.jpg']
print('left eyebrow')
print('load left_eyebrow')
for img in img_list4:
        input_img4=cv2.imread('./landmarksCK+1/train/left_eyebrow' +'/'+ img )
        input_img4=cv2.resize(input_img4, (70,70))
        input_img4=cv2.cvtColor(input_img4, cv2.COLOR_BGR2GRAY)
        img_data_list4.append(input_img4)

img_data4 = np.asarray(img_data_list4)
img_data4 = img_data4.astype('float32')
img_data4 = np.expand_dims(img_data4, -1)
'''
xtrain4, xtest4,ytrain4,ytest4 = train_test_split(img_data4, labels,test_size=0.2,shuffle=True)
xtrain4 = xtrain4.astype('float32')
xtest4 = xtest4.astype('float32')

xtrain4 /= 255
xtest4 /= 255

ytrain4 = to_categorical(ytrain4, 7)
ytest4 = to_categorical(ytest4, 7)'''


xtrain4 = img_data4
xtrain4 = xtrain4.astype('float32')
xtrain4 /= 255

img_data_list44 = []

img_list4 = [img_list for img_list in test_data_path4 if img_list[-4:]=='.png' or '.jpg']
for img in img_list4:
        input_img4=cv2.imread('./landmarksCK+1/test/left_eyebrow' + '/'+ img )
        input_img4=cv2.resize(input_img4, (70,70))
        input_img4=cv2.cvtColor(input_img4, cv2.COLOR_BGR2GRAY)
        img_data_list44.append(input_img4)

img_data4 = np.asarray(img_data_list44)
img_data4 = img_data4.astype('float32')
img_data4 = np.expand_dims(img_data4, -1)

xtest4 = img_data4
xtest4 = xtest4.astype('float32')
xtest4 /= 255 


img_list5 = [img_list for img_list in train_data_path5 if img_list[-4:]=='.png' or '.jpg']
print('mouth')
print('load mouth')
for img in img_list5:
        input_img5=cv2.imread('./landmarksCK+1/train/mouth' +'/'+ img )
        input_img5=cv2.resize(input_img5, (70,70))
        input_img5=cv2.cvtColor(input_img5, cv2.COLOR_BGR2GRAY)
        img_data_list5.append(input_img5)

img_data5 = np.asarray(img_data_list5)
img_data5 = img_data5.astype('float32')
img_data5 = np.expand_dims(img_data5, -1)

'''
xtrain5, xtest5,ytrain5,ytest5 = train_test_split(img_data5, labels,test_size=0.2,shuffle=True)
xtrain5 = xtrain5.astype('float32')
xtest5 = xtest5.astype('float32')

xtrain5 /= 255
xtest5 /= 255

ytrain5 = to_categorical(ytrain5, 7)
ytest5 = to_categorical(ytest5, 7) '''


xtrain5 = img_data5
xtrain5 = xtrain5.astype('float32')
xtrain5 /= 255

img_data_list55 = []

img_list5 = [img_list for img_list in test_data_path5 if img_list[-4:]=='.png' or '.jpg']
for img in img_list5:
        input_img5=cv2.imread('./landmarksCK+1/test/mouth' +'/'+ img )
        input_img5=cv2.resize(input_img5, (70,70))
        input_img5=cv2.cvtColor(input_img5, cv2.COLOR_BGR2GRAY)
        img_data_list55.append(input_img5)

img_data5 = np.asarray(img_data_list55)
img_data5 = img_data5.astype('float32')
img_data5 = np.expand_dims(img_data5, -1)

xtest5 = img_data5
xtest5 = xtest5.astype('float32')
xtest5 /= 255 

#nose dataset

img_list6 = [img_list for img_list in train_data_path6 if img_list[-4:]=='.png' or '.jpg']
print('nose')
print('load nose')
for img in img_list6:
        input_img6=cv2.imread('./landmarksCK+1/train/nose' +'/'+ img )
        input_img6=cv2.resize(input_img6, (70,70))
        input_img6=cv2.cvtColor(input_img6, cv2.COLOR_BGR2GRAY)
        img_data_list6.append(input_img6)

img_data6 = np.asarray(img_data_list6)
img_data6 = img_data6.astype('float32')
img_data6 = np.expand_dims(img_data6, -1)

'''
xtrain6, xtest6,ytrain6,ytest6 = train_test_split(img_data6, labels,test_size=0.2,shuffle=True)
xtrain6 = xtrain6.astype('float32')
xtest6 = xtest6.astype('float32')

xtrain6 /= 255
xtest6 /= 255

ytrain6 = to_categorical(ytrain6, 7)
ytest6 = to_categorical(ytest6, 7)'''

xtrain6 = img_data6
xtrain6 = xtrain6.astype('float32')
xtrain6 /= 255

img_data_list66 = []

img_list6 = [img_list for img_list in test_data_path6 if img_list[-4:]=='.png' or '.jpg']
for img in img_list6:
        input_img6=cv2.imread('./landmarksCK+1/test/nose' + '/'+ img )
        input_img6=cv2.resize(input_img6, (70,70))
        input_img6=cv2.cvtColor(input_img6, cv2.COLOR_BGR2GRAY)
        img_data_list66.append(input_img6)

img_data6 = np.asarray(img_data_list66)
img_data6 = img_data6.astype('float32')
img_data6 = np.expand_dims(img_data6, -1)

xtest6 = img_data6
xtest6 = xtest6.astype('float32')
xtest6 /= 255 

#jaw dataset

img_list7 = [img_list for img_list in train_data_path7 if img_list[-4:]=='.png' or '.jpg']
print('jaw')
print('load jaw')
for img in img_list7:
        input_img7=cv2.imread('./landmarksCK+1/train/jaw' +'/'+ img )
        input_img7=cv2.resize(input_img7, (70,70))
        input_img7=cv2.cvtColor(input_img7, cv2.COLOR_BGR2GRAY)
        img_data_list7.append(input_img7)

img_data7 = np.asarray(img_data_list7)
img_data7 = img_data7.astype('float32')
img_data7 = np.expand_dims(img_data7, -1)

'''
xtrain7, xtest7,ytrain7,ytest7 = train_test_split(img_data7, labels,test_size=0.2,shuffle=True)
xtrain7 = xtrain7.astype('float32')
xtest7 = xtest7.astype('float32')

xtrain7 /= 255
xtest7 /= 255

ytrain7 = to_categorical(ytrain7, 7)
ytest7 = to_categorical(ytest7, 7) '''


xtrain7 = img_data7
xtrain7 = xtrain7.astype('float32')
xtrain7 /= 255

img_data_list77 =[]

img_list7 = [img_list for img_list in test_data_path7 if img_list[-4:]=='.png' or '.jpg']
for img in img_list7:
        input_img7=cv2.imread('./landmarksCK+1/test/jaw' + '/'+ img )
        input_img7=cv2.resize(input_img7, (70,70))
        input_img7=cv2.cvtColor(input_img7, cv2.COLOR_BGR2GRAY)
        img_data_list77.append(input_img7)

img_data7 = np.asarray(img_data_list77)
img_data7 = img_data7.astype('float32')
img_data7 = np.expand_dims(img_data7, -1)

xtest7 = img_data7
xtest7 = xtest7.astype('float32')
xtest7 /= 255 

######################################## Model #############################################
# Create the model for right eye
'''right_eye = Sequential()

in1 = (None, None,1)
right_eye.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in1, batch_input_shape = (None, None, None, 1)))
right_eye.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
right_eye.add(MaxPooling2D(pool_size=(2, 2)))
right_eye.add(Dropout(0.20))'''
print(xtrain1.shape)
print(xtrain1.shape[1:])
inputs = Input(shape=xtrain1.shape[1:])
right_eye = inputs

#right_eye = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(right_eye)
#right_eye = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(right_eye)
#right_eye = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(right_eye)
#ight_eye = MaxPool2D(pool_size=(2, 2))(right_eye)
#right_eye = Dropout(rate=0.20)(right_eye)

right_eye = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(right_eye)
right_eye = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(right_eye)
right_eye = MaxPool2D(pool_size=(2, 2))(right_eye)
right_eye = Dropout(rate=0.20)(right_eye)
right_eye = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(right_eye)
right_eye = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(right_eye)
right_eye = MaxPool2D(pool_size=(2, 2))(right_eye)
right_eye = Dropout(rate=0.20)(right_eye)

right_eye= Conv2D(filters=128, kernel_size=(2,2), activation='relu')(right_eye)
right_eye = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(right_eye)
ight_eye = MaxPool2D(pool_size=(2, 2))(right_eye)
right_eye = Dropout(rate=0.20)(right_eye)

right_eye = Flatten()(right_eye)
right_eye = Dense(16, activation='relu')(right_eye)
right_eye = Dropout(rate=0.5)(right_eye)
'''right_eye = Dense(7, activation='relu')(right_eye) '''
right_eye = Model(inputs, right_eye) 

# Create the model for left eye
'''left_eye = Sequential()

in2 = (None, None,1)
left_eye.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in2, batch_input_shape = (None, None,None, 1)))
left_eye.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
left_eye.add(MaxPooling2D(pool_size=(2, 2)))
left_eye.add(Dropout(0.20))'''
inputs = Input(shape=xtrain2.shape[1:])
left_eye = inputs

'''left_eye = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(left_eye)
left_eye = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(left_eye)
#left_eye = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(left_eye)
left_eye = MaxPool2D(pool_size=(2, 2))(left_eye)
left_eye = Dropout(rate=0.20)(left_eye)'''

left_eye = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(left_eye)
left_eye = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(left_eye)
left_eye = MaxPool2D(pool_size=(2, 2))(left_eye)
left_eye = Dropout(rate=0.20)(left_eye)
left_eye = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(left_eye)
left_eye = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(left_eye)
left_eye = MaxPool2D(pool_size=(2, 2))(left_eye)
left_eye = Dropout(rate=0.20)(left_eye)

left_eye= Conv2D(filters=128, kernel_size=(2,2), activation='relu')(left_eye)
left_eye = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(left_eye)
left_eye = MaxPool2D(pool_size=(2, 2))(left_eye)
left_eye = Dropout(rate=0.20)(left_eye)

left_eye = Flatten()(left_eye) 
left_eye = Dense(16, activation='relu')(left_eye)
left_eye = Dropout(rate=0.5)(left_eye)
'''left_eye = Dense(7, activation='relu')(left_eye) '''
left_eye = Model(inputs, left_eye)

# Create the model for right eyebrow
'''right_eyebrow = Sequential()

in3 = (None, None, 1)
right_eyebrow.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in3, batch_input_shape = (None, None, None, 1)))
right_eyebrow.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
right_eyebrow.add(MaxPooling2D(pool_size=(2, 2)))
right_eyebrow.add(Dropout(0.20))'''
inputs = Input(shape=xtrain3.shape[1:])
right_eyebrow = inputs

'''#right_eyebrow = Conv2D(filters=16, kernel_size=(7, 7), activation='relu')(right_eyebrow)
right_eyebrow = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(right_eyebrow)
right_eyebrow = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(right_eyebrow)
right_eyebrow = MaxPool2D(pool_size=(2, 2))(right_eyebrow)
right_eyebrow = Dropout(rate=0.20)(right_eyebrow)'''

right_eyebrow= Conv2D(filters=32, kernel_size=(5,5), activation='relu')(right_eyebrow)
right_eyebrow = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(right_eyebrow)
right_eyebrow = MaxPool2D(pool_size=(2, 2))(right_eyebrow)
right_eyebrow = Dropout(rate=0.20)(right_eyebrow)
right_eyebrow = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(right_eyebrow)
right_eyebrow = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(right_eyebrow)
right_eyebrow = MaxPool2D(pool_size=(2, 2))(right_eyebrow)
right_eyebrow = Dropout(rate=0.20)(right_eyebrow)

right_eyebrow= Conv2D(filters=128, kernel_size=(2,2), activation='relu')(right_eyebrow)
right_eyebrow = Conv2D(filters=128, kernel_size=(2,2), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(right_eyebrow)
right_eyebrow = MaxPool2D(pool_size=(2, 2))(right_eyebrow)
right_eyebrow = Dropout(rate=0.20)(right_eyebrow)

right_eyebrow = Flatten()(right_eyebrow)
right_eyebrow = Dense(16, activation='relu')(right_eyebrow)
right_eyebrow = Dropout(rate=0.5)(right_eyebrow)
'''right_eyebrow= Dense(7, activation='relu')(right_eyebrow) '''
right_eyebrow = Model(inputs, right_eyebrow) 


# Create the model for left eyebrow
'''left_eyebrow = Sequential()

in4 = (None, None, 1)
left_eyebrow.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in4, batch_input_shape = (None, None, None, 1)))
left_eyebrow.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
left_eyebrow.add(MaxPooling2D(pool_size=(2, 2)))
left_eyebrow.add(Dropout(0.20))'''

print(xtrain4.shape[1:])
inputs = Input(shape=xtrain4.shape[1:])
left_eyebrow = inputs
'''
#left_eyebrow = Conv2D(filters=16, kernel_size=(7, 7), activation='relu')(left_eyebrow)
left_eyebrow = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(left_eyebrow)
left_eyebrow = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(left_eyebrow)
left_eyebrow = MaxPool2D(pool_size=(2, 2))(left_eyebrow)
left_eyebrow = Dropout(rate=0.20)(left_eyebrow)'''

left_eyebrow= Conv2D(filters=32, kernel_size=(5,5), activation='relu')(left_eyebrow)
left_eyebrow = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(left_eyebrow)
left_eyebrow = MaxPool2D(pool_size=(2, 2))(left_eyebrow)
left_eyebrow = Dropout(rate=0.20)(left_eyebrow)
left_eyebrow = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(left_eyebrow)
left_eyebrow = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(left_eyebrow)
left_eyebrow = MaxPool2D(pool_size=(2, 2))(left_eyebrow)
left_eyebrow = Dropout(rate=0.20)(left_eyebrow)

left_eyebrow= Conv2D(filters=128, kernel_size=(2,2), activation='relu')(left_eyebrow)
left_eyebrow = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(left_eyebrow)
left_eyebrow = MaxPool2D(pool_size=(2, 2))(left_eyebrow)
left_eyebrow = Dropout(rate=0.20)(left_eyebrow)

left_eyebrow = Flatten()(left_eyebrow)
left_eyebrow = Dense(16, activation='relu')(left_eyebrow)
left_eyebrow = Dropout(rate=0.5)(left_eyebrow)
'''left_eyebrow= Dense(7, activation='relu')(left_eyebrow) '''
left_eyebrow = Model(inputs, left_eyebrow) 


# Create the model for mouth
mouth = Sequential()

#mouth.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in5))
#mouth.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#mouth.add(MaxPooling2D(pool_size=(2, 2), data_format=None))
#mouth.add(Dropout(0.20))
'''from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

mouth = Sequential()
mouth.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain5.shape[1:]))
mouth.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
mouth.add(MaxPool2D(pool_size=(2, 2)))
mouth.add(Dropout(rate=0.20))
mouth.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mouth.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mouth.add(MaxPool2D(pool_size=(2, 2)))
mouth.add(Dropout(rate=0.20))
mouth.add(Flatten())
mouth.add(Dense(256, activation='relu'))
#print(mouth)
mouth.add(Dropout(rate=0.5))
print(mouth)
mouth.add(Dense(7, activation='softmax'))'''

inputs = Input(shape=xtrain5.shape[1:])
# Create the model for mouth
mouth = inputs
'''
#mouth = Conv2D(filters=16, kernel_size=(7, 7), activation='relu')(mouth)
mouth = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(mouth)
mouth = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(mouth)
mouth = MaxPool2D(pool_size=(2, 2))(mouth)
mouth = Dropout(rate=0.20)(mouth)'''


mouth = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(mouth)
mouth = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(mouth)
mouth = MaxPool2D(pool_size=(2, 2))(mouth)
mouth = Dropout(rate=0.20)(mouth)

mouth = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(mouth)
mouth = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(mouth)
mouth = MaxPool2D(pool_size=(2, 2))(mouth)
mouth = Dropout(rate=0.20)(mouth)

mouth = Flatten()(mouth)
mouth = Dense(16, activation='relu')(mouth)
mouth = Dropout(rate=0.5)(mouth)
'''mouth = Dense(7, activation='relu')(mouth) '''
mouth = Model(inputs, mouth) 


'''
# Create the model for nose
nose = Sequential()

in6 = (None, None, 1)
nose.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in6, batch_input_shape = (None, None, None, 1)))
nose.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
nose.add(MaxPooling2D(pool_size=(2, 2)))
nose.add(Dropout(0.20))
'''

print(xtrain6.shape[1:])
# Create the model for nose
inputs = Input(shape=xtrain6.shape[1:])
nose = inputs
'''
#nose = Conv2D(filters=16, kernel_size=(7, 7), activation='relu')(nose)
nose = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(nose)
nose = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(nose)
nose = MaxPool2D(pool_size=(2, 2))(nose)
nose = Dropout(rate=0.20)(nose)'''


nose = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(nose)
nose = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(nose)
nose = MaxPool2D(pool_size=(2, 2))(nose)
nose = Dropout(rate=0.25)(nose)

nose = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(nose)
nose = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(nose)
nose = MaxPool2D(pool_size=(2, 2))(nose)
nose = Dropout(rate=0.25)(nose)

nose = Flatten()(nose)
nose = Dense(16, activation='relu')(nose)
nose = Dropout(rate=0.5)(nose)
'''nose = Dense(7, activation='relu')(nose) '''
nose = Model(inputs, nose) 


# Create the model for jaw
inputs = Input(shape=xtrain7.shape[1:])
jaw = inputs
'''
#jaw = Conv2D(filters=16, kernel_size=(7, 7), activation='relu')(jaw)
jaw = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(jaw)
jaw = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.20)(jaw)'''


jaw = Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(0.0005),  activation='relu')(jaw)
jaw = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.20)(jaw)

jaw= Conv2D(filters=128, kernel_size=(2,2), activation='relu')(jaw)
jaw = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.20)(jaw)

jaw = Flatten()(jaw)
jaw = Dense(18, activation='relu')(jaw)
jaw = Dropout(rate=0.5)(jaw)
'''jaw = Dense(7, activation='relu')(jaw)'''
jaw = Model(inputs, jaw)

'''
# Create the model for jaw
jaw = Sequential()
jaw.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain7.shape[1:]))
jaw.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
jaw.add(MaxPool2D(pool_size=(2, 2)))
jaw.add(Dropout(rate=0.20))
jaw.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
jaw.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
jaw.add(MaxPool2D(pool_size=(2, 2)))
jaw.add(Dropout(rate=0.20))
jaw.add(Flatten())
jaw.add(Dense(256, activation='relu'))
jaw.add(Dropout(rate=0.5))
jaw.add(Dense(7, activation='softmax'))
'''


################################Train Models#############################################


combinedInput = concatenate([right_eye.output, left_eye.output, right_eyebrow.output, left_eyebrow.output, mouth.output, nose.output, jaw.output])
x = Dense(64, activation="relu")(combinedInput)
'''x = Dropout(rate=0.20)(x)
x = Dense(1028, activation="relu")(x)
x = Dropout(rate=0.20)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(rate=0.20)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(rate=0.20)(x) '''
x = Dense(7, activation="softmax")(x)
if mode == "train":
        model = Model(inputs=[right_eye.input, left_eye.input, right_eyebrow.input, left_eyebrow.input, mouth.input, nose.input, jaw.input], outputs=x)

        '''datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
        )'''
        print('---------------------{}'.format(xtest5.shape))
        print('---------------------{}'.format(xtest7.shape))

        #we need only one ytrain and ytest so we must to organise our dataset
        model.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False),
                        metrics=['acc'])
        history = model.fit([xtrain1, xtrain2, xtrain3, xtrain4, xtrain5, xtrain6, xtrain7], ytrain, 
        validation_data=([xtest1, xtest2, xtest3, xtest4, xtest5, xtest6, xtest7], ytest), 
        epochs=200, batch_size=32)



        #on plot les resultats pour vérifier les performances du réseau
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(len(acc))
        
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        
        plt.figure()
        
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
        plt.show()

        #save the model
        model.save('projetModel22.h5')
        model.save_weights('model_weight22.h5')

        
