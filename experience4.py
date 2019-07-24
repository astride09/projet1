from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from charger_les_données import load_fer2013, preprocess_input
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model



# les paramètres 
batch_size_train = 100
batch_size_test = 10
num_epochs = 50
input_shape = (48, 48, 3)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50


# data generator de keras pour la génération de données
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True, data_format=None)


# loading dataset
faces, emotions = load_fer2013()
faces = cv2.cvtColor(faces, cv2.COLOR_GRAY2RGB)
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)


# load model
model = load_model('transferEmotion.h5')
# summarize model.
model.summary()

#on met trainable à false pour tout les noeuds sauf les 4 derniers  
for layer in model.layers[:-4]:
    layer.trainable = False

# crée un nouveau model
new_model = model.Sequential()
 
# ajouter le modèle de base à new_model
new_model.add(model)
 
# on ajoute une une couche entièrement connectée suivie d'une couche softmax avec 7 sorties pour les 7 émotions repertoriées dans fer2013
new_model.add(layers.Flatten())
new_model.add(layers.Dense(1024, activation='relu'))
new_model.add(layers.Dropout(0.5))
new_model.add(layers.Dense(7, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
new_model.summary()

#compiler le model
new_model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

#former le model
history = new_model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size_train),
                        steps_per_epoch=len(xtrain) / batch_size_train,
                        epochs=num_epochs, verbose=1,
                        validation_data=(xtest,ytest),
                        validation_steps=len(xtest)/batch_size_test)

#sauvegarde le model
new_model.save('transferEmotionGan.h5')


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
