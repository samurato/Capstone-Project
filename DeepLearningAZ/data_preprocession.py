#Data Preprocessing 
#import The libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import datasets
datasets = pd.read_csv('https://raw.githubusercontent.com/EKami/deep_learning_A-Z/master/Get_the_machine_learning_basics/Data_Preprocessing_Template/Data.csv')

X= datasets.iloc[:,:-1].values
Y = datasets.iloc[:, -1].values

### Lets fill the datas with no values for e.g. Array indices no. 4 , Germany has no Salary and Indicies no. 6 Spain has not age data.
## First import Sci-kit learn
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])





### Encoding the Categorical data using labelencoder
####Label Encoder changes values to numerical values
###but first lets import labelencoders from Sci-kit learn

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
##Doing this will create a numerical values and create confusion ##that France is better that spain is better than France and ###Germany and Germany is better than France
##Which is not the case here.

#LEts use Dummy Encoding here to fix this issues
from sklearn.preprocessing import OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


######Model Making
###Spliting the datasets in to training sets
##import the librarirs that does it. Cross validation library
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_
