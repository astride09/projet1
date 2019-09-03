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
from keras import initializers
from keras import regularizers
import csv

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
a = ap.parse_args()
mode = a.mode 


label = np.ones(17166, dtype='int64')
with open('jaw_label.csv', 'r') as csvfile:
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
                label[k] = 0
        for k in r1:
                label[k] = 1
        for k in r2:
                label[k] = 2
        for k in r3:
                label[k] = 3
        for k in r4:
                label[k] = 4
        for k in r5:
                label[k] = 5
        for k in r6:
                label[k] = 6
print(label.shape)
"""
labels = np.ones((7700),dtype='int64')
labels[0:1099]=0 #2848
labels[1100:2199]=1 #679
labels[2200:3299]=2 #940
labels[3300:4399]=3 #960
labels[4400:5499]=4 #1372
labels[5500:6599]=5 #715
labels[5600:7699]=6 #586
print(labels.shape)
print(label) """
train_data_path7 = os.listdir('/content/drive/My Drive/test2/jaw')
test_data_path7 = os.listdir('/content/drive/My Drive/test2/jaw')

img_data_list, img_data_list1, img_data_list2, img_data_list3, img_data_list4, img_data_list5, img_data_list6, img_data_list7, img_data_list8 = [], [], [], [], [], [], [], [], []



#jaw dataset
'''for dataset in train_data_path7:
    img_list7=os.listdir('./landmarksCK+/jaw'+'/'+ dataset)
    img_list7 = [img_list for img_list in img_list7 if img_list[-4:]=='.png' or '.jpg']
    print('jaw')'''
for img in train_data_path7:
        input_img7=cv2.imread('/content/drive/My Drive/test2/jaw' + '/'+ img )
        input_img7=cv2.resize(input_img7, (70,70))
        input_img7=cv2.cvtColor(input_img7, cv2.COLOR_BGR2GRAY)
        img_data_list7.append(input_img7)

img_data7 = np.asarray(img_data_list7)
img_data7 = img_data7.astype('float32')
img_data7 = np.expand_dims(img_data7, -1)

xtrain7, xtest7,ytrain7,ytest7 = train_test_split(img_data7, label,test_size=0.2,shuffle=True)
xtrain7 = xtrain7.astype('float32')
xtest7 = xtest7.astype('float32')

xtrain7 /= 255
xtest7 /= 255

ytrain7 = to_categorical(ytrain7, 7)
ytest7 = to_categorical(ytest7, 7)


# Create the model for right eye
'''jaw = Sequential()

in1 = (None, None,1)
jaw.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in1, batch_input_shape = (None, None, None, 1)))
jaw.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
jaw.add(MaxPooling2D(pool_size=(2, 2)))
jaw.add(Dropout(0.25))'''
print(ytrain7.shape)
print(xtrain7.shape[1:])

# Create the model for jaw
inputs = Input(shape=xtrain7.shape[1:])
jaw = inputs
'''
#jaw = Conv2D(filters=16, kernel_size=(7, 7), activation='relu')(jaw)
jaw = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(jaw)
jaw = Conv2D(filters=16, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0005), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.20)(jaw)'''


jaw = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(jaw)
jaw = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.5)(jaw)

jaw= Conv2D(filters=128, kernel_size=(3,3), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.5)(jaw)

jaw = Flatten()(jaw)
jaw = Dense(10, activation='relu')(jaw)
jaw = Dropout(rate = 0.5)(jaw)
jaw = Dense(7, activation='softmax')(jaw)
jaw = Model(inputs, jaw)

''' inputs = Input(shape=xtrain1.shape[1:])
jaw = inputs

jaw = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(jaw)
jaw = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(jaw)
jaw = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.2)(jaw)
jaw = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(jaw)
jaw = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(jaw)
jaw = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.2)(jaw)

jaw = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(jaw)
jaw = Conv2D(filters=128, kernel_size=(2,2), activation='relu')(jaw)
jaw = Conv2D(filters=128, kernel_size=(2, 2), activation='relu')(jaw)
jaw = MaxPool2D(pool_size=(2, 2))(jaw)
jaw = Dropout(rate=0.2)(jaw)
jaw = Flatten()(jaw)

jaw = Dense(32, activation='relu')(jaw)
jaw = Dense(32, activation='relu')(jaw)
jaw = Dense(7, activation='softmax')(jaw)
jaw = Model(inputs, jaw)'''
import csv
import keras
if mode == "train":
        es_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=3)
        #we need only one ytrain and ytest so we must to organise our dataset
        jaw.compile(loss='categorical_crossentropy',
                                optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False),
                                metrics=['acc'])
        history = jaw.fit(xtrain7, ytrain7, callbacks=[es_callback],
                validation_data=(xtest7, ytest7), 
                epochs=200, batch_size=16)

         
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

elif mode == "test":
        '''title=["neutral7", "angry7", "sad7", "surprise7", "happy7", "fear7", "disgust7"]
        with open ('data.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([title])
        jaw.load_weights('model_weight7.h5')
        i=0
        for im in xtest7: 
                
                im = np.expand_dims(im, 0)
                prediction = jaw.predict(im)
                maxindex = int(np.argmax(prediction))
                prediction = prediction[0]
                                
                i = i+1
                with open ('data.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([prediction])'''

        
        for y in ytrain7:
                print(y)
