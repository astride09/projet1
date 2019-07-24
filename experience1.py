import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, BatchNormalization, Reshape
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
from pylab import rcParams
from PIL import Image
rcParams['figure.figsize'] = 20, 10


from sklearn.utils import shuffle

import keras

from keras.utils import np_utils

from keras import backend as K

class GAN():
    def __init__(self):
        self.img_rows=256
        self.img_cols=256
        self.channels=1
        self.num_epoch=10
        self.img_data_list=[]
        self.num_classes = 7
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def preprocess_input(self, x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def load_lfw(self):
        data_path = './BD/lfw'
        data_dir_list = os.listdir(data_path)
        label_list_name=[]
        for dataset in data_dir_list:
            img_list = os.listdir(data_path+'/'+ dataset)
            for img in img_list:
                label_list_name.append(dataset)
                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize=cv2.resize(input_img,(256,256))
                self.img_data_list.append(input_img_resize)

        img_data = np.asarray(self.img_data_list)
        img_data = img_data.astype('float32')
        img_data = np.expand_dims(img_data, -1)
        img_data.shape
        return label_list_name, img_data
                



    def load_jaffe(self):
        data_path = './jaffe/'
        data_dir_list = os.listdir(data_path)
        for dataset in data_dir_list:
            img_list=os.listdir(data_path+'/'+ dataset)
            print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
            for img in img_list:
                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize=cv2.resize(input_img,(256,256))
                self.img_data_list.append(input_img_resize)
                
        img_data = np.asarray(self.img_data_list)
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

    def getLabel(self, id):
        return ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE'][id]



    def train(self, epochs, batch_size=128, save_interval=1):

        # Load the dataset

        faces, labels = self.load_jaffe()
        # convert class labels to on-hot encoding# conve 
        labels = np_utils.to_categorical(labels, self.num_classes)
        X_train, xtest,ytrain,ytest = train_test_split(faces, labels, test_size=0.2,shuffle=True)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            
            self.save_imgs(epoch)

    def save_imgs(self, epoch):

        noise = np.random.normal(0, 1, (1, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        plt.savefig("images/"+"_"+str(epoch)+".png", bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=50, batch_size=32)