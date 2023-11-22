'''
    Scipt to test Neural Network implementation
    Created: 23/03/2023
    Author: Scott Taylor

    Robert Gordon University

    Achieved 90.54 training acc, iters=10000, Lr:0.15

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from NeuralNetwork import NueralNet

def ReLU(Z):
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    return Z > 0

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)

def init_params(size):
    W1 = np.random.rand(10,size)              #this may be just rand (15:46 in vid)
    b1 = np.random.rand(10,1)                #appaently the subtraction also breaks shit!
    W2 = np.random.rand(4,10)     
    b2 = np.random.rand(4,1)        
    return W1,b1,W2,b2

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m
    A1 = ReLU(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 #10,m
    A2 = softmax(Z2) #10,m
    return Z1, A1, Z2, A2

def one_Hot (Y):
    '''onehot code the labels'''

    onehot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_y = onehot_encoder.fit_transform(Y.reshape(-1,1))
    one_hot_y = one_hot_y.T     #transpose the matrix
    #print(one_hot_y.shape)

    return one_hot_y


def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    one_hot_Y = one_Hot(Y)
    #print(one_hot_Y)
    #print(one_hot_Y.shape)
    #print(np.sum(one_hot_Y))


    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (4,1))

    #W1 = W1 - alpha * dW1
    #b1 = b1 - alpha * db2
    #W2 = W2 - alpha * dW2
    #b2 = b2 - alpha * db2


    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, alpha, iterations):
    size , m = X.shape

    print(size)
    print("m",m)

    W1, b1, W2, b2 = init_params(size)
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)   

        if (i+1) % int(iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')
    return W1, b1, W2, b2

def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index,X, Y, W1, b1, W2, b2):
    # None => cree un nouvel axe de dimension 1, cela a pour effet de transposer X[:,index] qui un np.array de dimension 1 (ligne) et qui devient un vecteur (colonne)
    #  ce qui correspond bien a ce qui est demande par make_predictions qui attend une matrice dont les colonnes sont les pixels de l'image, la on donne une seule colonne
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

##############################################################################################

#handy little trick!
#for i in range(1, 25):
#    print("'US{}',".format(i), sep =' ', end='',flush=True)

headers = ['US1','US2','US3','US4','US5','US6','US7','US8','US9',
           'US10','US11','US12','US13','US14','US15','US16','US17',
           'US18','US19','US20','US21','US22','US23','US24','Class']

df = pd.read_csv('SensorReadings24.txt', names=headers)

print(df)

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


#df = pd.get_dummies(df, columns=['Class'])
#df.dropna(inplace=True)
#print(df.head())


#split data into train and test set 
#First, split the data into 4 groups depending on o/p

df_MF   = df[df['Class'] == 'Move-Forward']
df_SIRT = df[df['Class'] == 'Slight-Right-Turn']
df_SILT = df[df['Class'] == 'Slight-Left-Turn']
df_SHRT = df[df['Class'] == 'Sharp-Right-Turn']

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

    X = df.drop(columns=['Class'])
    y_label = df['Class'].values.reshape(X.shape[0],1)  #reshape values to a 1-D array
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
                      

#at this point the data is now in the correct format!

print(New_Xtest)

#convert the data to numpy arrays

New_Xtest = np.array(New_Xtest).T
New_Xtrain = np.array(New_Xtrain).T
New_ytest = np.array(New_ytest).T
New_ytrain = np.array(New_ytrain).T


print(f"Shape of train set is {New_Xtrain}")
print(f"Shape of test set is {New_Xtest.shape}")
print(f"Shape of train label is {New_ytrain.shape}")
print(f"Shape of test labels is {New_ytest.shape}")

#np.savetxt("NEW_Xtrain.csv", New_Xtrain, delimiter=',')


W1, b1, W2, b2 = gradient_descent(New_Xtrain, New_ytrain, 0.15, 100)

#from keras.layers import Dense
#from keras.models import Sequential
#
#model = Sequential()
#model.add(Dense(10, activation='relu', input_dim=24))
#model.add(Dense(4, activation='softmax'))
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#print(model.summary())
#
#New_ytrain = one_Hot(New_ytrain).T
#print(f"Shape of train label is {New_ytrain.shape}")
#hist = model.fit(New_Xtrain, New_ytrain, epochs=100)


