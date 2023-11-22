''''
    Script to test neural network implementation
    Created: 31/03/2023
    Author: Scott Taylor

    Robert Gordons University

    Accuracy Achieved: 91.53%
    Iterations: 10,000
    Learning Rate: 0.15

'''

import pandas as pd
import numpy as np
#from sklearn.preprocessing import OneHotEncoder

#custom librarys
from NeuralNet import NeuralNetwork
from DataProcessing import Processing

#headers for the df
headers = ['US1','US2','US3','US4','US5','US6','US7','US8','US9',
           'US10','US11','US12','US13','US14','US15','US16','US17',
           'US18','US19','US20','US21','US22','US23','US24','Class']

df = pd.read_csv('SensorReadings24.txt', names=headers)


#normalise the data
for col in df.columns:
    try:
        df[col] = Processing.normalise(df[col])
    except TypeError:
        continue

#one hot encode the labels
df = pd.get_dummies(df, columns=['Class'])
df.dropna(inplace=True)


#split the data into train and test sets
proc = Processing()
y_test, y_train, X_test, X_train = proc.train_test_split(df)

nn = NeuralNetwork(iterations=10000, learning_rate=0.05)

#Train the network and return the optimised weights & biases
W1, b1, W2, b2 = nn.gradient_descent(X_train, y_train)

#use the test data to verify the networks effectivness
test_pred, mse = nn.make_prediction(X_test,y_test)


print(f"Test Accuracy: {nn.get_accuracy(test_pred, y_test):.3%}")
print(f"MSE: {mse}")