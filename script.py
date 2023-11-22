'''
    Scipt to test Neural Network implementation
    Created: 23/03/2023
    Author: Scott Taylor

    Robert Gordon University

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from NeuralNetwork import NueralNet

#handy little trick!
#for i in range(1, 25):
#    print("'US{}',".format(i), sep =' ', end='',flush=True)

headers = ['US1','US2','US3','US4','US5','US6','US7','US8','US9',
           'US10','US11','US12','US13','US14','US15','US16','US17',
           'US18','US19','US20','US21','US22','US23','US24','Class']

df = pd.read_csv('SensorReadings24.txt', names=headers)

#print(df)

#print(df.isna().sum())      #check for missing values
#print(df.dtypes)            #check to ensure all features are numeric

#normalise the columns 

def min_max_scalling(series):
    '''Normalise the data in a column'''
    return (series - (series.min()*0.8)) / ((series.max()*1.2) - (series.min()*0.8))

for col in df.columns:
    try:
        df[col] = min_max_scalling(df[col])
    except TypeError:
        continue

print(df.head())

#encode the output data
#Move-Forward       = 1000 (8)
#Slight-Right-Turn  = 0100 (4)  
#Sharp-Right-Turn   = 0010 (2)
#Slight-Left-Turn   = 0001 (1)

#had to use the bin function here as you cant have integers with leading 0's
#df['Class'] = df['Class'].replace('Move-Forward',bin(8))
#df['Class'] = df['Class'].replace('Slight-Right-Turn',bin(4))
#df['Class'] = df['Class'].replace('Sharp-Right-Turn',bin(2))
#df['Class'] = df['Class'].replace('Slight-Left-Turn',bin(1))

df = pd.get_dummies(df, columns=['Class'])
df.dropna(inplace=True)
print(df.head())


#split data into train and test set 
#First, split the data into 4 groups depending on o/p

df_MF   = df[df['Class_Move-Forward'] == 1]
df_SIRT = df[df['Class_Slight-Right-Turn'] == 1]
df_SHRT = df[df['Class_Sharp-Right-Turn'] == 1]
df_SILT = df[df['Class_Slight-Left-Turn'] ==1]

allData = []
allData.append(df_MF)
allData.append(df_SHRT)
allData.append(df_SILT)
allData.append(df_SIRT)

print(df_MF)
print(df_SHRT)
print(df_SILT)
print(df_SIRT)

def formatData(df):
    '''perform nessecary formatting on data'''

    #stratify=y parameter allows you to select similar proportions for each dataset (https://training.atmosera.com/binary-classification/) about 75% down the page

    X = df.drop(columns=['Class_Move-Forward','Class_Slight-Right-Turn','Class_Sharp-Right-Turn','Class_Slight-Left-Turn'])
    y_label = df['Class_Move-Forward','Class_Slight-Right-Turn','Class_Sharp-Right-Turn','Class_Slight-Left-Turn'].values.reshape(X.shape[0],1)  #reshape values to a 1-D array
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.3, random_state=2)

    return Xtrain, Xtest, ytrain, ytest

New_Xtrain = pd.DataFrame()
New_Xtest = pd.DataFrame()
New_ytrain = np.empty([1,1])
New_ytest = np.empty([1,1])

for item in allData:
    Xtrain, Xtest, ytrain, ytest = formatData(item)
    New_Xtrain = pd.concat([New_Xtrain,Xtrain])
    New_Xtest = pd.concat([New_Xtest,Xtest])
    New_ytrain = np.concatenate([New_ytrain,ytrain])
    New_ytest = np.concatenate([New_ytest,ytest])

#delete the value that appears as a result of the empty placeholder array
New_ytest = np.delete(New_ytest,0,0)    #specify the axis as zero to stop the array becoming 'flattened'
New_ytrain = np.delete(New_ytrain,0,0)
                      
print(f"Shape of train set is {New_Xtrain.shape}")
print(f"Shape of test set is {New_Xtest.shape}")
print(f"Shape of train label is {New_ytrain.shape}")
print(f"Shape of test labels is {New_ytest.shape}")

#at this point the data is now in the correct format!

#initilise the NN and train the model 
nn = NueralNet()
nn.fit(Xtrain,ytrain)

#plot the results
nn.plot_loss()

#show the accuracy 
train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

print("Train Accuracy is {}".format(nn.acc(ytrain, train_pred)))
print("Test Accuracy is {}".format(nn.acc(ytest, test_pred)))
