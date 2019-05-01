#Convolutional Neural Networks 

#Part -1 

# Images are 3d array having values of colour of RGb, And Pixel values
# Keras library

#Data Prepeocessing
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Installing the CNN
classifier = Sequential()

#Building of CNN     
#    Convolution=> MaxPooliong => Flatten => FullConnection.

#Step 1 Convolution
classifier.add(Convolution2D(32,3,3, input_shape =(64,64,3), activation='relu'))

#Step2 Maxpooling Increases the effeciency without compromising the accracy
classifier.add(MaxPooling2D(pool_size=(2,2)))
#------------for Better results---------

#add multilayer Neural Network and second layer of maxpooling
classifier.add(Convolution2D(32,3,3,  activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))





#Step 4 Flattenning of image Putting the matrix into a single vector which goes to fully connected layers
classifier.add(Flatten())

#Step 5  Full Connection 
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))
#If you have more than binary output we use softmax rather than sigmoid


#Compile the Neural nets using sotachistic gradiant decent= adam as optimiser, For binary use binary cross entropy if not binary use categorical cross entropy 

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#-------------------------------
#Part2 Fitting the CNN to Images
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#lesser the target size the better is the result but is resource heavy

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

##### Making a new Predictions

import numpy as np

from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
#Convert image to array
test_imge = image.img_to_array(test_image)
#add the third dimension
test_image = np.expand_dims(test_image, axis = 0)
#Prediction
resutl = classifier.predict(test_image)
##We get value of 0 and 1
#Map the Training sets 
training_set.class_indices
























