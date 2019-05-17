#RNN Tutorials
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#PArt 1 Data Preprocessing
#Import Training Sets
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

training_set = dataset_train.iloc[:,1:2].values

#Feature Scaling normalisation and sanderisation 
from sklearn.preprocessing import MinMaxScaler
#Normalisation of Data between 0 and 1
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

##!!Specifing Data Structure what the RNN needs to remember when Predicting the Stock Price.
#Creating a data strucuture with 60 timestamps and 1 output (t+1)
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    #trainging_set_scaled[before 60 to 0 for 0th coloumn]
    y_train.append(training_set_scaled[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping the Data by adding new dimensions or indicators.
#TO align with the 3d  input shape https://keras.io/layers/recurrent/
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# Part 2 Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
#Calling it regressor as we are predicting the continous output a we are doing similar to regression so we call it regressor

regressor  = Sequential()

#Addint the first LSTM layer and some dropouts




#Three important ipput sfor LSTM, Unit which us number of LSTM Cells we want to have , Return Sequence to be true as we want to return the stacked sequences and the input shapes is the shape of the input shape of X Train

regressor.add(LSTM(units =50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#Adding the units to be 50 will gives us high dimentionality and better results, Adding the Dropout to 0.2 means that 20% of the neurons are dropped.



####Adding the Second LSTM and Dropout for regularisation
regressor.add(LSTM(units =50, return_sequences= True))
regressor.add(Dropout(0.2))



###Adding the Thrid LSTM layer and some Dropout regularisation
regressor.add(LSTM(units =50, return_sequences= True))
regressor.add(Dropout(0.2))


###Adding Fourth LSTM and some Dropout regularisation
regressor.add(LSTM(units =50))
regressor.add(Dropout(0.2))

###### Adding the Output Layer
#Because we need fully connected layer in output we use dense and the unit as 1 as the output is only one dimension, Output corresponds to output which is the dimension output which is one.
regressor.add(Dense(units = 1))

#Compiling the RNN with right optimiser and right loss function

regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')


#Fitting the RNN with the Training Sets

# Below will link and execute the training sets
# INput the X train and y Train as the inputs epoch can be any number but 100 was found to be better and the batch size of 32 which mean it will perform the back propagation
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#Part 3 Maing predictions and Visualisation
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_set = dataset_test.iloc[:,1:2].values
#Getting the real Stock PRice of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values

#Fising the shape problem by scaling be keeping the test value as they ate 
inputs = inputs.reshape(-1,1)                       
inputs  = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#Visualiosation of results
plt.plot(real_stock_set, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price Value')
plt.legend()
plt.show()