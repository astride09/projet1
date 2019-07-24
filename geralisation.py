from keras.models import load_model
from charger_les_données import load_fer2013, preprocess_input
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import confusion_matrix 


# loading dataset
faces, emotions = load_fer2013()
faces = cv2.cvtColor(faces, cv2.COLOR_GRAY2RGB)
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)


# load model
model = load_model('transferEmotion.h5')

eval_model = model.evaluate (xtrain, ytrain) 
eval_model

ypred = model.predict (xtest) 
ypred = (ypred> 0.5)

cm = confusion_matrix (ytest, ypred) 
print (cm)