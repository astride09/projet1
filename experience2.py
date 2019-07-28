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
#data_path1 = os.listdir('./landmarks/lfw/right_eye')
#data_path2 = os.listdir('./landmarks/lfw/left_eye')
#data_path3 = os.listdir('./landmarks/lfw/right_eyebrow')
#data_path4 = os.listdir ('./landmarks/lfw/left_eyebrow')
data_path5 = os.listdir('./landmark/mouth')
#data_path6 = os.listdir('./landmarks/lfw/nose')
data_path7 = os.listdir('./landmark/jaw')

img_data_list1, img_data_list2, img_data_list3, img_data_list4, img_data_list5, img_data_list6, img_data_list7 = [], [], [], [], [], [], []
'''for dataset in data_path1:

    img_list1=os.listdir('./landmarks/lfw/right_eye'+'/'+ dataset)
    for img in img_list1:
        input_img1=cv2.imread('./landmarks/lfw/right_eye' + '/'+ dataset + '/'+ img )
        input_img1=cv2.resize(input_img1, (250,100))
        img_data_list1.append(input_img1)

img_data1 = np.asarray(img_data_list1)
img_data1 = img_data1.astype('float32')
img_data1 = np.expand_dims(img_data1, -1)
img_data1.shape
xtrain1, xtest1,ytrain1,ytest1 = train_test_split(img_data1, labels,test_size=0.2,shuffle=True)

for dataset in data_path2:
    img_list2=os.listdir('./landmarks/lfw/left_eye'+'/'+ dataset)
    for img in img_list2:
        input_img2=cv2.imread('./landmarks/lfw/left_eye' + '/'+ dataset + '/'+ img )
        input_img2=cv2.resize(input_img2,(250,100))
        img_data_list2.append(input_img2)

img_data2 = np.asarray(img_data_list2)
img_data2 = img_data2.astype('float32')
img_data2 = np.expand_dims(img_data2, -1)
img_data2.shape
xtrain2, xtest2,ytrain2,ytest2 = train_test_split(img_data2, labels_list,test_size=0.2,shuffle=True)

for dataset in data_path3:
    img_list3=os.listdir('./landmarks/lfw/right_eyebrow'+'/'+ dataset)
    for img in img_list3:
        input_img3=cv2.imread('./landmarks/lfw/right_eyebrow' + '/'+ dataset + '/'+ img )
        input_img3=cv2.resize(input_img3,(250,100))
        img_data_list3.append(input_img3)

img_data3 = np.asarray(img_data_list3)
img_data3 = img_data3.astype('float32')
img_data3 = np.expand_dims(img_data3, -1)
img_data3.shape
xtrain3, xtest3,ytrain3,ytest3 = train_test_split(img_data3, labels_list,test_size=0.2,shuffle=True)

for dataset in data_path4:
    img_list4=os.listdir('./landmarks/lfw/left_eyebrow'+'/'+ dataset)
    for img in img_list4:
        input_img4=cv2.imread('./landmarks/lfw/left_eyebrow' + '/'+ dataset + '/'+ img )
        input_img4=cv2.resize(input_img4,(250,100))
        img_data_list4.append(input_img4)

img_data4 = np.asarray(img_data_list4)
img_data4 = img_data4.astype('float32')
img_data4 = np.expand_dims(img_data4, -1)
img_data4.shape
xtrain4, xtest4,ytrain4,ytest4 = train_test_split(img_data4, labels_list,test_size=0.2,shuffle=True)
'''
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
'''
for dataset in data_path6:
    img_list6=os.listdir('./landmarks/lfw/nose'+'/'+ dataset)
    for img in img_list6:
        input_img6=cv2.imread('./landmarks/lfw/nose' + '/'+ dataset + '/'+ img )
        input_img6=cv2.resize(input_img6,(250,100))
        img_data_list6.append(input_img6)

img_data6 = np.asarray(img_data_list6)
img_data6 = img_data6.astype('float32')
img_data6 = np.expand_dims(img_data6, -1)
img_data6.shape
xtrain6, xtest6,ytrain6,ytest6 = train_test_split(img_data6, labels_list,test_size=0.2,shuffle=True)
'''
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
'''right_eye = Sequential()

in1 = (None, None,1)
right_eye.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in1, batch_input_shape = (None, None, None, 1)))
right_eye.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
right_eye.add(MaxPooling2D(pool_size=(2, 2)))
right_eye.add(Dropout(0.25))

# Create the model for leftt eye
left_eye = Sequential()

in2 = (None, None,1)
left_eye.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in2, batch_input_shape = (None, None,None, 1)))
left_eye.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
left_eye.add(MaxPooling2D(pool_size=(2, 2)))
left_eye.add(Dropout(0.25))

# Create the model for right eyebrow
right_eyebrow = Sequential()

in3 = (None, None, 1)
right_eyebrow.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in3, batch_input_shape = (None, None, None, 1)))
right_eyebrow.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
right_eyebrow.add(MaxPooling2D(pool_size=(2, 2)))
right_eyebrow.add(Dropout(0.25))


# Create the model for leftt eyebrow
left_eyebrow = Sequential()

in4 = (None, None, 1)
left_eyebrow.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in4, batch_input_shape = (None, None, None, 1)))
left_eyebrow.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
left_eyebrow.add(MaxPooling2D(pool_size=(2, 2)))
left_eyebrow.add(Dropout(0.25))


'''
# Create the model for mouth
mouth = Sequential()

#mouth.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in5))
#mouth.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#mouth.add(MaxPooling2D(pool_size=(2, 2), data_format=None))
#mouth.add(Dropout(0.25))
from keras.models import Sequential
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
print(mouth)
mouth.add(Dropout(rate=0.5))
print(mouth)
mouth.add(Dense(7, activation='softmax'))



'''
# Create the model for nose
nose = Sequential()

in6 = (None, None, 1)
nose.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in6, batch_input_shape = (None, None, None, 1)))
nose.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
nose.add(MaxPooling2D(pool_size=(2, 2)))
nose.add(Dropout(0.25))

'''
# Create the model for jaw
jaw = Sequential()
jaw.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=xtrain5.shape[1:]))
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






'''###########################################Model Final
model_final_concat = Concatenate([right_eye, left_eye, right_eyebrow, left_eyebrow, mouth, nose, jaw])
model_final_dense_1 = Dense(len(img_data1) + len(img_data2)+len(img_data3)+len(img_data4)+len(img_data5)+len(img_data6)+len(img_data7), activation='softmax')(model_final_concat)

model = Model(inputs=[in1, in2, in3, in4, in5, in6, in7], outputs=model_final_dense_1)


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history = model.fit([xtrain1, xtrain2, xtrain3, xtrain4, xtrain5, xtrain6, xtrain7], [ytrain1, ytrain2, ytrain3, ytrain4, 
ytrain4, ytrain5, ytrain6, ytrain7],
          batch_size=32, nb_epoch=30, verbose=1)



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