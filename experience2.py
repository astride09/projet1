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

optimizer = Adam(0.0002, 0.5)

train_labels = np.ones((149),dtype='int64')
train_labels[0:20]=0 #21
train_labels[21:40]=1 #20
train_labels[41:62]=2 #22
train_labels[63:84]=3 #22
train_labels[85:105]=4 #21
train_labels[106:127]=5 #22
train_labels[128:]=6 #21
names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']

ytrain = train_labels
ytrain = to_categorical(ytrain, 7)


test_labels = np.ones((64),dtype='int64')
test_labels[0:8]=0 #9
test_labels[9:17]=1 #9
test_labels[18:27]=2 #10
test_labels[28:36]=3 #9
test_labels[37:45]=4 #9
test_labels[46:54]=5 #9
test_labels[55:]=6 #9

ytest = test_labels
ytest = to_categorical(ytest, 7)
names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']

#load dataset
train_data_path1 = os.listdir('/content/drive/My Drive/jaffeLandmark/train/right_eye')
test_data_path1 = os.listdir('/content/drive/My Drive/jaffeLandmark/test/right_eye')

train_data_path2 = os.listdir('/content/drive/My Drive/jaffeLandmark/train/left_eye')
test_data_path2 = os.listdir('/content/drive/My Drive/jaffeLandmark/test/left_eye')

train_data_path3 = os.listdir('/content/drive/My Drive/jaffeLandmark/train/right_eyebrow')
test_data_path3 = os.listdir('/content/drive/My Drive/jaffeLandmark/test/right_eyebrow')

train_data_path4 = os.listdir('/content/drive/My Drive/jaffeLandmark/train/left_eyebrow')
test_data_path4 = os.listdir('/content/drive/My Drive/jaffeLandmark/test/left_eyebrow')

train_data_path5 = os.listdir('/content/drive/My Drive/jaffeLandmark/train/mouth')
test_data_path5 = os.listdir('/content/drive/My Drive/jaffeLandmark/test/mouth')

train_data_path6 = os.listdir('/content/drive/My Drive/jaffeLandmark/train/nose')
test_data_path6 = os.listdir('/content/drive/My Drive/jaffeLandmark/test/nose')

train_data_path7 = os.listdir('/content/drive/My Drive/jaffeLandmark/train/jaw')
test_data_path7 = os.listdir('/content/drive/My Drive/jaffeLandmark/test/jaw')



img_data_list, img_data_list1, img_data_list2, img_data_list3, img_data_list4, img_data_list5, img_data_list6, img_data_list7 = [], [], [], [], [], [], [], []

'''def load_jaffe():
        data_path = '/content/drive/My Drive/jaffe'
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
for dataset in train_data_path1:
    
    #print(data_path5)
    #print(data_path5)
    img_list1=os.listdir('/content/drive/My Drive/jaffeLandmark/train/right_eye'+'/'+ dataset)
    img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.jpg']
    #print(img_list5)
    for img in img_list1:
        input_img1=cv2.imread('/content/drive/My Drive/jaffeLandmark/train/right_eye' + '/'+ dataset + '/'+ img )
        input_img1=cv2.resize(input_img1, (150,150))
        input_img1=cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
        img_data_list1.append(input_img1)

img_data1 = np.asarray(img_data_list1)
img_data1 = img_data1.astype('float32')
img_data1 = np.expand_dims(img_data1, -1)
#print(img_data5.shape)
'''xtrain1, xtest1,ytrain1,ytest1 = train_test_split(img_data1, labels,test_size=0.2,shuffle=True)
xtrain1 = xtrain1.astype('float32')
xtest1 = xtest1.astype('float32')

xtrain1 /= 255
xtest1 /= 255

ytrain1 = to_categorical(ytrain1, 7)
ytest1 = to_categorical(ytest1, 7)'''

xtrain1 = img_data1
xtrain1 = xtrain1.astype('float32')
xtrain1 /= 255

img_data_list11 = []
for dataset in test_data_path1:
    
    #print(data_path5)
    #print(data_path5)
    img_list1=os.listdir('/content/drive/My Drive/jaffeLandmark/test/right_eye'+'/'+ dataset)
    img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.jpg']
    #print(img_list5)
    for img in img_list1:
        input_img1=cv2.imread('/content/drive/My Drive/jaffeLandmark/test/right_eye' + '/'+ dataset + '/'+ img )
        input_img1=cv2.resize(input_img1, (150,150))
        input_img1=cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
        img_data_list11.append(input_img1)

img_data1 = np.asarray(img_data_list11)
img_data1 = img_data1.astype('float32')
img_data1 = np.expand_dims(img_data1, -1)

xtest1 = img_data1
xtest1 = xtest1.astype('float32')
xtest1 /= 255


for dataset in train_data_path2:
    
    #print(data_path5)
    #print(data_path5)
    img_list2 = os.listdir('/content/drive/My Drive/jaffeLandmark/train/left_eye'+'/'+ dataset)
    img_list2 = [img_list for img_list in img_list2 if img_list[-4:]=='.jpg']
    #print(img_list5)
    for img in img_list2:
        input_img2=cv2.imread('/content/drive/My Drive/jaffeLandmark/train/left_eye' + '/'+ dataset + '/'+ img )
        input_img2=cv2.resize(input_img2, (150,150))
        input_img2=cv2.cvtColor(input_img2, cv2.COLOR_BGR2GRAY)
        img_data_list2.append(input_img2)

img_data2 = np.asarray(img_data_list2)
img_data2 = img_data2.astype('float32')
img_data2 = np.expand_dims(img_data2, -1)

xtrain2 = img_data2
xtrain2 = xtrain2.astype('float32')
xtrain2 /= 255

img_data_list22 = []
for dataset in test_data_path2:
    
    #print(data_path5)
    #print(data_path5)
    img_list2 = os.listdir('/content/drive/My Drive/jaffeLandmark/test/left_eye'+'/'+ dataset)
    img_list2 = [img_list for img_list in img_list2 if img_list[-4:]=='.jpg']
    #print(img_list5)
    for img in img_list2:
        input_img2=cv2.imread('/content/drive/My Drive/jaffeLandmark/test/left_eye' + '/'+ dataset + '/'+ img )
        input_img2=cv2.resize(input_img2, (150,150))
        input_img2=cv2.cvtColor(input_img2, cv2.COLOR_BGR2GRAY)
        img_data_list22.append(input_img2)

img_data2 = np.asarray(img_data_list22)
img_data2 = img_data2.astype('float32')
img_data2 = np.expand_dims(img_data2, -1)

xtest2 = img_data2
xtest2 = xtest2.astype('float32')
xtest2 /= 255

for dataset in train_data_path3:
    
    #print(data_path5)
    #print(data_path5)
    img_list3=os.listdir('/content/drive/My Drive/jaffeLandmark/train/right_eyebrow'+'/'+ dataset)
    img_list3 = [img_list for img_list in img_list3 if img_list[-4:]=='.jpg']
    #print(img_list5)
    for img in img_list3:
        input_img3=cv2.imread('/content/drive/My Drive/jaffeLandmark/train/right_eyebrow' + '/'+ dataset + '/'+ img )
        input_img3=cv2.resize(input_img3, (150,150))
        input_img3=cv2.cvtColor(input_img3, cv2.COLOR_BGR2GRAY)
        img_data_list3.append(input_img3)

img_data3 = np.asarray(img_data_list3)
img_data3 = img_data3.astype('float32')
img_data3 = np.expand_dims(img_data3, -1)

xtrain3 = img_data3
xtrain3 = xtrain3.astype('float32')
xtrain3 /= 255

img_data_list33 = []
for dataset in test_data_path3:
    
    #print(data_path5)
    #print(data_path5)
    img_list3=os.listdir('/content/drive/My Drive/jaffeLandmark/test/right_eyebrow'+'/'+ dataset)
    img_list3 = [img_list for img_list in img_list3 if img_list[-4:]=='.jpg']
    #print(img_list5)
    for img in img_list3:
        input_img3=cv2.imread('/content/drive/My Drive/jaffeLandmark/test/right_eyebrow' + '/'+ dataset + '/'+ img )
        input_img3=cv2.resize(input_img3, (150,150))
        input_img3=cv2.cvtColor(input_img3, cv2.COLOR_BGR2GRAY)
        img_data_list33.append(input_img3)

img_data3 = np.asarray(img_data_list33)
img_data3 = img_data3.astype('float32')
img_data3 = np.expand_dims(img_data3, -1)

xtest3 = img_data3
xtest3 = xtest3.astype('float32')
xtest3 /= 255

#left_eyebrow dataset
for dataset in train_data_path4:
    img_list4=os.listdir('/content/drive/My Drive/jaffeLandmark/train/left_eyebrow'+'/'+ dataset)
    img_list4 = [img_list for img_list in img_list4 if img_list[-4:]=='.jpg']
    for img in img_list4:
        input_img4=cv2.imread('/content/drive/My Drive/jaffeLandmark/train/left_eyebrow' + '/'+ dataset + '/'+ img )
        input_img4=cv2.resize(input_img4, (150,150))
        input_img4=cv2.cvtColor(input_img4, cv2.COLOR_BGR2GRAY)
        img_data_list4.append(input_img4)

img_data4 = np.asarray(img_data_list4)
img_data4 = img_data4.astype('float32')
img_data4 = np.expand_dims(img_data4, -1)

xtrain4 = img_data4
xtrain4 = xtrain4.astype('float32')
xtrain4 /= 255

img_data_list44 = []
for dataset in test_data_path4:
    img_list4=os.listdir('/content/drive/My Drive/jaffeLandmark/test/left_eyebrow'+'/'+ dataset)
    img_list4 = [img_list for img_list in img_list4 if img_list[-4:]=='.jpg']
    for img in img_list4:
        input_img4=cv2.imread('/content/drive/My Drive/jaffeLandmark/test/left_eyebrow' + '/'+ dataset + '/'+ img )
        input_img4=cv2.resize(input_img4, (150,150))
        input_img4=cv2.cvtColor(input_img4, cv2.COLOR_BGR2GRAY)
        img_data_list44.append(input_img4)

img_data4 = np.asarray(img_data_list44)
img_data4 = img_data4.astype('float32')
img_data4 = np.expand_dims(img_data4, -1)

xtest4 = img_data4
xtest4 = xtest4.astype('float32')
xtest4 /= 255

for dataset in train_data_path5:
    img_list5=os.listdir('/content/drive/My Drive/jaffeLandmark/train/mouth'+'/'+ dataset)
    img_list5 = [img_list for img_list in img_list5 if img_list[-4:]=='.jpg']
    for img in img_list5:
        input_img5=cv2.imread('/content/drive/My Drive/jaffeLandmark/train/mouth' + '/'+ dataset + '/'+ img )
        input_img5=cv2.resize(input_img5, (150,150))
        input_img5=cv2.cvtColor(input_img5, cv2.COLOR_BGR2GRAY)
        img_data_list5.append(input_img5)

img_data5 = np.asarray(img_data_list5)
img_data5 = img_data5.astype('float32')
img_data5 = np.expand_dims(img_data5, -1)

xtrain5 = img_data5
xtrain5 = xtrain5.astype('float32')
xtrain5 /= 255

img_data_list55 = []
for dataset in test_data_path5:
    img_list5=os.listdir('/content/drive/My Drive/jaffeLandmark/test/mouth'+'/'+ dataset)
    img_list5 = [img_list for img_list in img_list5 if img_list[-4:]=='.jpg']
    for img in img_list5:
        input_img5=cv2.imread('/content/drive/My Drive/jaffeLandmark/test/mouth' + '/'+ dataset + '/'+ img )
        input_img5=cv2.resize(input_img5, (150,150))
        input_img5=cv2.cvtColor(input_img5, cv2.COLOR_BGR2GRAY)
        img_data_list55.append(input_img5)

img_data5 = np.asarray(img_data_list55)
img_data5 = img_data5.astype('float32')
img_data5 = np.expand_dims(img_data5, -1)

xtest5 = img_data5
xtest5 = xtest5.astype('float32')
xtest5 /= 255

#nose dataset
for dataset in train_data_path6:
    img_list6=os.listdir('/content/drive/My Drive/jaffeLandmark/train/nose'+'/'+ dataset)
    img_list6 = [img_list for img_list in img_list6 if img_list[-4:]=='.jpg']
    for img in img_list6:
        input_img6=cv2.imread('/content/drive/My Drive/jaffeLandmark/train/nose' + '/'+ dataset + '/'+ img )
        input_img6=cv2.resize(input_img6, (150,150))
        input_img6=cv2.cvtColor(input_img6, cv2.COLOR_BGR2GRAY)
        img_data_list6.append(input_img6)

img_data6 = np.asarray(img_data_list6)
img_data6 = img_data6.astype('float32')
img_data6 = np.expand_dims(img_data6, -1)

xtrain6 = img_data6
xtrain6 = xtrain6.astype('float32')
xtrain6 /= 255

img_data_list66 = []
for dataset in test_data_path6:
    img_list6=os.listdir('/content/drive/My Drive/jaffeLandmark/test/nose'+'/'+ dataset)
    img_list6 = [img_list for img_list in img_list6 if img_list[-4:]=='.jpg']
    for img in img_list6:
        input_img6=cv2.imread('/content/drive/My Drive/jaffeLandmark/test/nose' + '/'+ dataset + '/'+ img )
        input_img6=cv2.resize(input_img6, (150,150))
        input_img6=cv2.cvtColor(input_img6, cv2.COLOR_BGR2GRAY)
        img_data_list66.append(input_img6)

img_data6 = np.asarray(img_data_list66)
img_data6 = img_data6.astype('float32')
img_data6 = np.expand_dims(img_data6, -1)

xtest6 = img_data6
xtest6 = xtest6.astype('float32')
xtest6 /= 255

#jaw dataset
for dataset in train_data_path7:
    img_list7=os.listdir('/content/drive/My Drive/jaffeLandmark/train/jaw'+'/'+ dataset)
    img_list7 = [img_list for img_list in img_list7 if img_list[-4:]=='.jpg']
    for img in img_list7:
        input_img7=cv2.imread('/content/drive/My Drive/jaffeLandmark/train/jaw' + '/'+ dataset + '/'+ img )
        input_img7=cv2.resize(input_img7, (150,150))
        input_img7=cv2.cvtColor(input_img7, cv2.COLOR_BGR2GRAY)
        img_data_list7.append(input_img7)

img_data7 = np.asarray(img_data_list7)
img_data7 = img_data7.astype('float32')
img_data7 = np.expand_dims(img_data7, -1)

xtrain7 = img_data7
xtrain7 = xtrain7.astype('float32')
xtrain7 /= 255

img_data_list77 =[]
for dataset in test_data_path7:
    img_list7=os.listdir('/content/drive/My Drive/jaffeLandmark/test/jaw'+'/'+ dataset)
    img_list7 = [img_list for img_list in img_list7 if img_list[-4:]=='.jpg']
    for img in img_list7:
        input_img7=cv2.imread('/content/drive/My Drive/jaffeLandmark/test/jaw' + '/'+ dataset + '/'+ img )
        input_img7=cv2.resize(input_img7, (150,150))
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
right_eye.add(Dropout(0.25))'''
inputs = Input(shape=xtrain1.shape[1:])
right_eye = inputs
right_eye = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(right_eye)
right_eye = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(right_eye)
right_eye = MaxPool2D(pool_size=(2, 2))(right_eye)
right_eye = Dropout(rate=0.25)(right_eye)
right_eye = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(right_eye)
right_eye = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(right_eye)
right_eye = MaxPool2D(pool_size=(2, 2))(right_eye)
right_eye = Dropout(rate=0.25)(right_eye)
right_eye = Flatten()(right_eye)
right_eye = Dense(256, activation='relu')(right_eye)
right_eye = Dropout(rate=0.5)(right_eye)
right_eye = Dense(7, activation='softmax')(right_eye)
right_eye = Model(inputs, right_eye)

# Create the model for leftt eye
'''left_eye = Sequential()

in2 = (None, None,1)
left_eye.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in2, batch_input_shape = (None, None,None, 1)))
left_eye.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
left_eye.add(MaxPooling2D(pool_size=(2, 2)))
left_eye.add(Dropout(0.25))'''
inputs = Input(shape=xtrain2.shape[1:])
left_eye = inputs
left_eye = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(left_eye)
left_eye = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(left_eye)
left_eye = MaxPool2D(pool_size=(2, 2))(left_eye)
left_eye = Dropout(rate=0.25)(left_eye)
left_eye = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(left_eye)
left_eye = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(left_eye)
left_eye = MaxPool2D(pool_size=(2, 2))(left_eye)
left_eye = Dropout(rate=0.25)(left_eye)
left_eye = Flatten()(left_eye)
left_eye = Dense(256, activation='relu')(left_eye)
left_eye = Dropout(rate=0.5)(left_eye)
left_eye = Dense(7, activation='softmax')(left_eye)
left_eye = Model(inputs, left_eye)

# Create the model for right eyebrow
'''right_eyebrow = Sequential()

in3 = (None, None, 1)
right_eyebrow.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in3, batch_input_shape = (None, None, None, 1)))
right_eyebrow.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
right_eyebrow.add(MaxPooling2D(pool_size=(2, 2)))
right_eyebrow.add(Dropout(0.25))'''
inputs = Input(shape=xtrain3.shape[1:])
right_eyebrow = inputs
right_eyebrow= Conv2D(filters=32, kernel_size=(5,5), activation='relu')(right_eyebrow)
right_eyebrow = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(right_eyebrow)
right_eyebrow = MaxPool2D(pool_size=(2, 2))(right_eyebrow)
right_eyebrow = Dropout(rate=0.25)(right_eyebrow)
right_eyebrow = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(right_eyebrow)
right_eyebrow = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(right_eyebrow)
right_eyebrow = MaxPool2D(pool_size=(2, 2))(right_eyebrow)
right_eyebrow = Dropout(rate=0.25)(right_eyebrow)
right_eyebrow = Flatten()(right_eyebrow)
right_eyebrow = Dense(256, activation='relu')(right_eyebrow)
right_eyebrow = Dropout(rate=0.5)(right_eyebrow)
right_eyebrow= Dense(7, activation='softmax')(right_eyebrow)
right_eyebrow = Model(inputs, right_eyebrow)


# Create the model for leftt eyebrow
'''left_eyebrow = Sequential()

in4 = (None, None, 1)
left_eyebrow.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in4, batch_input_shape = (None, None, None, 1)))
left_eyebrow.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
left_eyebrow.add(MaxPooling2D(pool_size=(2, 2)))
left_eyebrow.add(Dropout(0.25))'''


inputs = Input(shape=xtrain4.shape[1:])
leftt_eyebrow = inputs
leftt_eyebrow= Conv2D(filters=32, kernel_size=(5,5), activation='relu')(leftt_eyebrow)
leftt_eyebrow = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(leftt_eyebrow)
leftt_eyebrow = MaxPool2D(pool_size=(2, 2))(leftt_eyebrow)
leftt_eyebrow = Dropout(rate=0.25)(leftt_eyebrow)
leftt_eyebrow = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(leftt_eyebrow)
leftt_eyebrow = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(leftt_eyebrow)
leftt_eyebrow = MaxPool2D(pool_size=(2, 2))(leftt_eyebrow)
leftt_eyebrow = Dropout(rate=0.25)(leftt_eyebrow)
leftt_eyebrow = Flatten()(leftt_eyebrow)
leftt_eyebrow = Dense(256, activation='relu')(leftt_eyebrow)
leftt_eyebrow = Dropout(rate=0.5)(leftt_eyebrow)
leftt_eyebrow= Dense(7, activation='softmax')(leftt_eyebrow)
leftt_eyebrow = Model(inputs, leftt_eyebrow)


# Create the model for mouth
mouth = Sequential()

#mouth.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in5))
#mouth.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#mouth.add(MaxPooling2D(pool_size=(2, 2), data_format=None))
#mouth.add(Dropout(0.25))
'''from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

mouth = Sequential()
mouth.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain5.shape[1:]))
mouth.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
mouth.add(MaxPool2D(pool_size=(2, 2)))
mouth.add(Dropout(rate=0.25))
mouth.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mouth.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mouth.add(MaxPool2D(pool_size=(2, 2)))
mouth.add(Dropout(rate=0.25))
mouth.add(Flatten())
mouth.add(Dense(256, activation='relu'))
#print(mouth)
mouth.add(Dropout(rate=0.5))
print(mouth)
mouth.add(Dense(7, activation='softmax'))'''

inputs = Input(shape=xtrain5.shape[1:])
# Create the model for mouth
mouth = inputs

#mouth.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in5))
#mouth.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#mouth.add(MaxPooling2D(pool_size=(2, 2), data_format=None))
#mouth.add(Dropout(0.25)


mouth = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(mouth)
mouth = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(mouth)
mouth = MaxPool2D(pool_size=(2, 2))(mouth)
mouth = Dropout(rate=0.25)(mouth)
mouth = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(mouth)
mouth = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(mouth)
mouth = MaxPool2D(pool_size=(2, 2))(mouth)
mouth = Dropout(rate=0.25)(mouth)
mouth = Flatten()(mouth)
mouth = Dense(256, activation='relu')(mouth)
mouth = Dropout(rate=0.5)(mouth)
mouth = Dense(7, activation='softmax')(mouth)
mouth = Model(inputs, mouth)


'''
# Create the model for nose
nose = Sequential()

in6 = (None, None, 1)
nose.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in6, batch_input_shape = (None, None, None, 1)))
nose.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
nose.add(MaxPooling2D(pool_size=(2, 2)))
nose.add(Dropout(0.25))
'''

print(xtrain6.shape[1:])
# Create the model for nose
inputs = Input(shape=xtrain6.shape[1:])
nose = inputs
nose= Conv2D(filters=32, kernel_size=(5,5), activation='relu')(nose)
nose = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(nose)
nose = MaxPool2D(pool_size=(2, 2))(nose)
nose = Dropout(rate=0.25)(nose)
nose = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(nose)
nose = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(nose)
nose = MaxPool2D(pool_size=(2, 2))(nose)
nose = Dropout(rate=0.25)(nose)
nose = Flatten()(nose)
nose = Dense(256, activation='relu')(nose)
nose = Dropout(rate=0.5)(nose)
nose = Dense(7, activation='softmax')(nose)
nose = Model(inputs, nose)


# Create the model for jaw
inputs = Input(shape=xtrain7.shape[1:])
jaw = inputs
jaw= Conv2D(filters=32, kernel_size=(5,5), activation='relu')(jaw)
jaw = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.25)(jaw)
jaw = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(jaw)
jaw = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.25)(jaw)
jaw = Flatten()(jaw)
jaw = Dense(256, activation='relu')(jaw)
jaw = Dropout(rate=0.5)(jaw)
jaw = Dense(7, activation='softmax')(jaw)
jaw = Model(inputs, jaw)

'''
# Create the model for jaw
jaw = Sequential()
jaw.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain7.shape[1:]))
jaw.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
jaw.add(MaxPool2D(pool_size=(2, 2)))
jaw.add(Dropout(rate=0.25))
jaw.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
jaw.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
jaw.add(MaxPool2D(pool_size=(2, 2)))
jaw.add(Dropout(rate=0.25))
jaw.add(Flatten())
jaw.add(Dense(256, activation='relu'))
jaw.add(Dropout(rate=0.5))
jaw.add(Dense(7, activation='softmax'))
'''


################################Train Models#############################################
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)

combinedInput = concatenate([right_eye.output, left_eye.output, right_eyebrow.output, leftt_eyebrow.output, mouth.output, nose.output, jaw.output])
x = Dense(32, activation="softmax")(combinedInput)
x = Dense(7, activation="softmax")(x)

model = Model(inputs=[right_eye.input, left_eye.input, right_eyebrow.input, leftt_eyebrow.input, mouth.input, nose.input, jaw.input], outputs=x)

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
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc'])
history1 = model.fit([xtrain1, xtrain2, xtrain3, xtrain4, xtrain5, xtrain6, xtrain7], ytrain, 
validation_data=([xtest1, xtest2, xtest3, xtest4, xtest5, xtest6, xtest7], ytest), 
epochs=2000, batch_size=2)



'''mouth.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history1 = mouth.fit_generator(datagen.flow(xtrain5, ytrain5, batch_size=2), epochs=25,
                              validation_data=(xtest5, ytest5), steps_per_epoch=xtrain5.shape[0]//2)

jaw.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history2 = jaw.fit_generator(datagen.flow(xtrain7, ytrain7, batch_size=2), epochs=25,
                              validation_data=(xtest7, ytest7), steps_per_epoch=xtrain7.shape[0]//2)

'''

'''###########################################Model Final
result = jaw*0.6 + mouth*0.4
model_final_dense_1 = Dense(7, activation='softmax')

model = Model(inputs=result, outputs=model_final_dense_1)


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history = model.fit_generator(datagen.flow(X_train, ytrain, batch_size=2), epochs=25,
                              validation_data=(xtest, ytest), steps_per_epoch=xtrain7.shape[0]//2)
'''

'''
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

#sauvegarde le model
model.save('projetModel.h5')
'''


'''
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Concatenate, Reshape
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
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical

optimizer = Adam(0.0002, 0.5)

labels = np.ones((213),dtype='int64')

labels[0:29]=0 #30
labels[30:58]=1 #29
labels[59:90]=2 #32
labels[91:121]=3 #31
labels[122:151]=4 #30
labels[152:182]=5 #31
labels[183:]=6 #30
print(labels)
names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']

#load dataset
data_path1 = os.listdir('./landmarks/lfw/right_eye')
data_path2 = os.listdir('./landmarks/lfw/left_eye')
data_path3 = os.listdir('./landmarks/lfw/right_eyebrow')
data_path4 = os.listdir ('./landmarks/lfw/left_eyebrow')
data_path5 = os.listdir('./landmark/mouth')
data_path6 = os.listdir('./landmarks/lfw/nose')
data_path7 = os.listdir('./landmark/jaw')

img_data_list1, img_data_list2, img_data_list3, img_data_list4, img_data_list5, img_data_list6, img_data_list7 = [], [], [], [], [], [], []

#right_eye dataset
for dataset in data_path1:
    img_list1=os.listdir('/content/projet1/landmark/right_eye'+'/'+ dataset)
    img_list1 = [img_list for img_list in img_list1 if img_list[-4:]=='.jpg']
    for img in img_list1:
        input_img1=cv2.imread('/content/projet1/landmark/right_eye' + '/'+ dataset + '/'+ img )
        input_img1=cv2.resize(input_img1, (150,150))
        input_img1=cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
        img_data_list1.append(input_img1)

img_data1 = np.asarray(img_data_list1)
img_data1 = img_data1.astype('float32')
img_data1 = np.expand_dims(img_data1, -1)
xtrain1, xtest1,ytrain1,ytest1 = train_test_split(img_data1, labels,test_size=0.2,shuffle=True)

xtrain1 = xtrain1.astype('float32')
xtest1 = xtest1.astype('float32')

xtrain1 /= 255
xtest1 /= 255

from keras.utils import to_categorical

ytrain1 = to_categorical(ytrain1, 7)
ytest1 = to_categorical(ytest1, 7)


#left_eye dataset
for dataset in data_path2:
    img_list2=os.listdir('/content/projet1/landmark/left_eye'+'/'+ dataset)
    img_list2 = [img_list for img_list in img_list2 if img_list[-4:]=='.jpg']
    #print(img_list7)
    for img in img_list2:
        input_img2=cv2.imread('/content/projet1/landmark/left_eye' + '/'+ dataset + '/'+ img )
        input_img2=cv2.resize(input_img2, (150,150))
        input_img2=cv2.cvtColor(input_img2, cv2.COLOR_BGR2GRAY)
        img_data_list2.append(input_img2)

img_data2 = np.asarray(img_data_list2)
img_data2 = img_data2.astype('float32')
img_data2 = np.expand_dims(img_data2, -1)
xtrain2, xtest2,ytrain2,ytest2 = train_test_split(img_data2, labels,test_size=0.2,shuffle=True)

xtrain2 = xtrain2.astype('float32')
xtest2 = xtest2.astype('float32')

xtrain2 /= 255
xtest2 /= 255

from keras.utils import to_categorical

ytrain2 = to_categorical(ytrain2, 7)
ytest2 = to_categorical(ytest2, 7)

#right_eyebrow dataset
for dataset in data_path3:
    img_list3=os.listdir('/content/projet1/landmark/right_eyebrow'+'/'+ dataset)
    img_list3 = [img_list for img_list in img_list3 if img_list[-4:]=='.jpg']
    for img in img_list3:
        input_img3=cv2.imread('/content/projet1/landmark/right_eyebrow' + '/'+ dataset + '/'+ img )
        input_img3=cv2.resize(input_img3, (150,150))
        input_img3=cv2.cvtColor(input_img3, cv2.COLOR_BGR2GRAY)
        img_data_list3.append(input_img3)

img_data3 = np.asarray(img_data_list3)
img_data3 = img_data3.astype('float32')
img_data3 = np.expand_dims(img_data3, -1)
#print(img_data7.shape)
xtrain3, xtest3,ytrain3,ytest3 = train_test_split(img_data3, labels,test_size=0.2,shuffle=True)

xtrain3 = xtrain3.astype('float32')
xtest3 = xtest3.astype('float32')

xtrain3 /= 255
xtest3 /= 255

from keras.utils import to_categorical

ytrain3 = to_categorical(ytrain3, 7)
ytest3 = to_categorical(ytest3, 7)


#left_eyebrow dataset
for dataset in data_path4:
    img_list4=os.listdir('/content/projet1/landmark/left_eyebrow'+'/'+ dataset)
    img_list4 = [img_list for img_list in img_list4 if img_list[-4:]=='.jpg']
    #print(img_list7)
    for img in img_list4:
        input_img4=cv2.imread('/content/projet1/landmark/left_eyebrow' + '/'+ dataset + '/'+ img )
        input_img4=cv2.resize(input_img4, (150,150))
        input_img4=cv2.cvtColor(input_img4, cv2.COLOR_BGR2GRAY)
        img_data_list4.append(input_img4)

img_data4 = np.asarray(img_data_list4)
img_data4 = img_data4.astype('float32')
img_data4 = np.expand_dims(img_data4, -1)
#print(img_data7.shape)
xtrain4, xtest4,ytrain4,ytest4 = train_test_split(img_data4, labels,test_size=0.2,shuffle=True)

xtrain4 = xtrain4.astype('float32')
xtest4 = xtest4.astype('float32')

xtrain4 /= 255
xtest4 /= 255

ytrain4 = to_categorical(ytrain4, 7)
ytest4 = to_categorical(ytest4, 7)


for dataset in data_path5:

    #print(data_path5)
    #print(data_path5)
    img_list5=os.listdir('/content/projet1/landmark/mouth'+'/'+ dataset)
    img_list5 = [img_list for img_list in img_list5 if img_list[-4:]=='.jpg']
    #print(img_list5)
    for img in img_list5:
        input_img5=cv2.imread('/content/projet1/landmark/mouth' + '/'+ dataset + '/'+ img )
        input_img5=cv2.resize(input_img5, (150,150))
        input_img5=cv2.cvtColor(input_img5, cv2.COLOR_BGR2GRAY)
        img_data_list5.append(input_img5)

img_data5 = np.asarray(img_data_list5)
img_data5 = img_data5.astype('float32')
img_data5 = np.expand_dims(img_data5, -1)
#print(img_data5.shape)
xtrain5, xtest5,ytrain5,ytest5 = train_test_split(img_data5, labels,test_size=0.2,shuffle=True)
xtrain5 = xtrain5.astype('float32')
xtest5 = xtest5.astype('float32')

xtrain5 /= 255
xtest5 /= 255

from keras.utils import to_categorical

ytrain5 = to_categorical(ytrain5, 7)
ytest5 = to_categorical(ytest5, 7)
"""print(xtrain5)
print('-------------------------------------------------')
print(xtrain5.shape)
print('-------------------------------------------------')
print(ytrain5.shape)"""

for dataset in data_path6:
    img_list6=os.listdir('/content/projet1/landmark/nose'+'/'+ dataset)
    img_list6 = [img_list for img_list in img_list6 if img_list[-4:]=='.jpg']
    #print(img_list7)
    for img in img_list6:
        input_img6=cv2.imread('/content/projet1/landmark/nose' + '/'+ dataset + '/'+ img )
        input_img6=cv2.resize(input_img6, (150,150))
        input_img6=cv2.cvtColor(input_img6, cv2.COLOR_BGR2GRAY)
        img_data_list6.append(input_img6)

img_data6 = np.asarray(img_data_list6)
img_data6 = img_data6.astype('float32')
img_data6 = np.expand_dims(img_data6, -1)
#print(img_data7.shape)
xtrain6, xtest6,ytrain6,ytest6 = train_test_split(img_data6, labels,test_size=0.2,shuffle=True)

xtrain6 = xtrain6.astype('float32')
xtest6 = xtest6.astype('float32')

xtrain6 /= 255
xtest6 /= 255

from keras.utils import to_categorical

ytrain6 = to_categorical(ytrain6, 7)
ytest6 = to_categorical(ytest6, 7)

for dataset in data_path7:
    img_list7=os.listdir('/content/projet1/landmark/mouth'+'/'+ dataset)
    img_list7 = [img_list for img_list in img_list7 if img_list[-4:]=='.jpg']
    #print(img_list7)
    for img in img_list7:
        input_img7=cv2.imread('/content/projet1/landmark/mouth' + '/'+ dataset + '/'+ img )
        input_img7=cv2.resize(input_img7, (150,150))
        input_img7=cv2.cvtColor(input_img7, cv2.COLOR_BGR2GRAY)
        img_data_list7.append(input_img7)

img_data7 = np.asarray(img_data_list7)
img_data7 = img_data7.astype('float32')
img_data7 = np.expand_dims(img_data7, -1)
#print(img_data7.shape)
xtrain7, xtest7,ytrain7,ytest7 = train_test_split(img_data7, labels,test_size=0.2,shuffle=True)

xtrain7 = xtrain7.astype('float32')
xtest7 = xtest7.astype('float32')

xtrain7 /= 255
xtest7 /= 255

from keras.utils import to_categorical

ytrain7 = to_categorical(ytrain7, 7)
ytest7 = to_categorical(ytest7, 7)

######################################## Model #############################################
# Create the model for right eye
right_eye = Sequential()

right_eye.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain1.shape[1:]))
right_eye.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
right_eye.add(MaxPool2D(pool_size=(2, 2)))
right_eye.add(Dropout(rate=0.25))
right_eye.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
right_eye.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
right_eye.add(MaxPool2D(pool_size=(2, 2)))
right_eye.add(Dropout(rate=0.25))
right_eye.add(Flatten())
right_eye.add(Dense(256, activation='relu'))
right_eye.add(Dropout(rate=0.5))
right_eye.add(Dense(7, activation='softmax'))

# Create the model for leftt eye
left_eye = Sequential()

left_eye.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain2.shape[1:]))
left_eye.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
left_eye.add(MaxPool2D(pool_size=(2, 2)))
left_eye.add(Dropout(rate=0.25))
left_eye.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
left_eye.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
left_eye.add(MaxPool2D(pool_size=(2, 2)))
left_eye.add(Dropout(rate=0.25))
left_eye.add(Flatten())
left_eye.add(Dense(256, activation='relu'))
left_eye.add(Dropout(rate=0.5))
left_eye.add(Dense(7, activation='softmax'))


# Create the model for right eyebrow
right_eyebrow = Sequential()

right_eyebrow.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain3.shape[1:]))
right_eyebrow.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
right_eyebrow.add(MaxPool2D(pool_size=(2, 2)))
right_eyebrow.add(Dropout(rate=0.25))
right_eyebrow.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
right_eyebrow.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
right_eyebrow.add(MaxPool2D(pool_size=(2, 2)))
right_eyebrow.add(Dropout(rate=0.25))
right_eyebrow.add(Flatten())
right_eyebrow.add(Dense(256, activation='relu'))
right_eyebrow.add(Dropout(rate=0.5))
right_eyebrow.add(Dense(7, activation='softmax'))



# Create the model for leftt eyebrow
left_eyebrow = Sequential()

left_eyebrow.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain3.shape[1:]))
left_eyebrow.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
left_eyebrow.add(MaxPool2D(pool_size=(2, 2)))
left_eyebrow.add(Dropout(rate=0.25))
left_eyebrow.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
left_eyebrow.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
left_eyebrow.add(MaxPool2D(pool_size=(2, 2)))
left_eyebrow.add(Dropout(rate=0.25))
left_eyebrow.add(Flatten())
left_eyebrow.add(Dense(256, activation='relu'))
left_eyebrow.add(Dropout(rate=0.5))
left_eyebrow.add(Dense(7, activation='softmax'))


# Create the model for mouth
mouth = Sequential()

#mouth.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in5))
#mouth.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#mouth.add(MaxPooling2D(pool_size=(2, 2), data_format=None))
#mouth.add(Dropout(0.25))

mouth = Sequential()
mouth.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain5.shape[1:]))
mouth.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
mouth.add(MaxPool2D(pool_size=(2, 2)))
mouth.add(Dropout(rate=0.25))
mouth.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mouth.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mouth.add(MaxPool2D(pool_size=(2, 2)))
mouth.add(Dropout(rate=0.25))
mouth.add(Flatten())
mouth.add(Dense(256, activation='relu'))
print(mouth)
mouth.add(Dropout(rate=0.5))
print(mouth)
mouth.add(Dense(7, activation='softmax'))




# Create the model for nose
nose = Sequential()

nose.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain6.shape[1:]))
nose.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
nose.add(MaxPool2D(pool_size=(2, 2)))
nose.add(Dropout(rate=0.25))
nose.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
nose.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
nose.add(MaxPool2D(pool_size=(2, 2)))
nose.add(Dropout(rate=0.25))
nose.add(Flatten())
nose.add(Dense(256, activation='relu'))
nose.add(Dropout(rate=0.5))
nose.add(Dense(7, activation='softmax'))



# Create the model for jaw
jaw = Sequential()
jaw.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain7.shape[1:]))
jaw.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
jaw.add(MaxPool2D(pool_size=(2, 2)))
jaw.add(Dropout(rate=0.25))
jaw.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
jaw.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
jaw.add(MaxPool2D(pool_size=(2, 2)))
jaw.add(Dropout(rate=0.25))
jaw.add(Flatten())
jaw.add(Dense(256, activation='relu'))
jaw.add(Dropout(rate=0.5))
jaw.add(Dense(7, activation='softmax'))




################################Train Models#############################################
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
  rotation_range=10,
  zoom_range=0.1,
  width_shift_range=0.1,
  height_shift_range=0.1
)


mouth.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history1 = mouth.fit_generator(datagen.flow(xtrain5, ytrain5, batch_size=2), epochs=25,
                              validation_data=(xtest5, ytest5), steps_per_epoch=xtrain5.shape[0]//2)

jaw.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history2 = jaw.fit_generator(datagen.flow(xtrain7, ytrain7, batch_size=2), epochs=25,
                              validation_data=(xtest7, ytest7), steps_per_epoch=xtrain7.shape[0]//2)


nose.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history2 = nose.fit_generator(datagen.flow(xtrain6, ytrain6, batch_size=2), epochs=25,
                              validation_data=(xtest6, ytest6), steps_per_epoch=xtrain6.shape[0]//2)

right_eye.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history2 = right_eye.fit_generator(datagen.flow(xtrain1, ytrain1, batch_size=2), epochs=25,
                              validation_data=(xtest1, ytest1), steps_per_epoch=xtrain1.shape[0]//2)

left_eye.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history2 = left_eye.fit_generator(datagen.flow(xtrain2, ytrain2, batch_size=2), epochs=25,
                              validation_data=(xtest2, ytest2), steps_per_epoch=xtrain2.shape[0]//2)

right_eyebrow.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history2 = right_eyebrow.fit_generator(datagen.flow(xtrain3, ytrain3, batch_size=2), epochs=25,
                              validation_data=(xtest3, ytest3), steps_per_epoch=xtrain3.shape[0]//2)

left_eyebrow.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history2 = left_eyebrow.fit_generator(datagen.flow(xtrain4, ytrain4, batch_size=2), epochs=25,
                              validation_data=(xtest4, ytest4), steps_per_epoch=xtrain4.shape[0]//2)
'''