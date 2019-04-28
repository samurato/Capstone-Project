# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encode the categorical data 
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encode the countries to numbers
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Encpde the gender to numbers 
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Create dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#-*--------------------------
#

#Part 2 :- Making an ANN, Import Keras,  builds Ann with #Tensorflow backend
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout


#Initialising the ANN to perform classification
classifier= Sequential()

#Add Input layer and first hidden layer with dropouts
classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))
classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu'))
classifier.add(Dropout(p=0.1))
##Output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation ='sigmoid'))

#Compile ANN :-Applying Stocastic Gradinet Decent
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

 ##Fitting classifier in tthe training sets
classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


#Use our ANN model to predict if the customer with the following informations will leave the bank: 
#
#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000
#So should we say goodbye to that customer ?
#
#The solution is provided in the next Lecture but I strongly recommend that you try to solve it on your own.


homework_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
homework_pred = (homework_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#---------------------------------

#Tuning and Evlauating the ANN using sci-kit learn
#careful crossvalidation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

from keras.models import Sequential 
from keras.layers import Dense
def build_classifier():
   classifier= Sequential()
   classifier.add(Dense(units=6,kernel_initializer="uniform", activation = 'relu', input_dim = 11))
   classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu'))
   classifier.add(Dense(units=1, kernel_initializer="uniform", activation ='sigmoid'))
   classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
   return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
#in windows PC if it is not working use this 
if __name__ == "__main__":
    #Execute accuracies only if the Multicore processing is supported
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs= -1 )









# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_














