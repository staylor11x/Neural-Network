'''
    Scipt to test Neural Network implementation
    Created: 23/03/2023
    Author: Scott Taylor

    Robert Gordon University

    Achieved 89.31% Accuracy a=10000 Lr=0.15

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

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
    W1 = np.random.randn(10,24)/10  #-0.5              #this may be just rand (15:46 in vid)
    b1 = np.random.randn(10,1)   #-0.5                #appaently the subtraction also breaks shit!
    W2 = np.random.randn(4,10)/10   #-0.5     
    b2 = np.random.randn(4,1)    #-0.5        
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

    return one_hot_y


def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    #one_hot_Y = one_Hot(Y)
    #print(one_hot_Y)
    #print(one_hot_Y.shape)
    #print(np.sum(one_hot_Y))


    dZ2 = 2*(A2 - Y) #10,m
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

    return W1, b1, W2, b2

def get_predictions(A2):
    '''Find the location of the max value in the columns of the o/p matrix
        i.e. the value that the network had predicted
    '''
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    '''get the accuracy of the network
        -- get_predictions returns the index of the max values for the predicted values
        -- then we need to do the same for the actual values
    '''
    Y = np.argmax(Y, axis=0)                    #get the index of the predicted values (works as 1 represents predicted value)
    return np.sum(predictions == Y)/Y.size

def mean_sqr_error(Y_true, Y_pred):
    '''calculate the mean squared error of the network'''
    return np.square(np.subtract(Y_true,Y_pred)).mean()

def plot_error(mse, iterations):
    
    plt.scatter(iterations,mse)
    plt.xlabel('Iteration')
    plt.ylabel('mse')
    plt.title('MSE vs Epoch')
    plt.show()
    


def gradient_descent(X, Y, alpha, iterations):
    size , m = X.shape

    print(size)
    print("m",m)

    error = []
    iters = []

    W1, b1, W2, b2 = init_params(size)
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)   

        if (i+1) % int(iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')

            mse = mean_sqr_error(Y, A2)
            print(mse)

            error.append(mse)
            iters.append(i+1)
            
    print(error)
    print(iters)
    plot_error(error, iters)

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

    #current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

    #plt.gray()
    #plt.imshow(current_image, interpolation='nearest')
    #plt.show()

##############################################################################################

#handy little trick!
#for i in range(1, 25):
#    print("'US{}',".format(i), sep =' ', end='',flush=True)

headers = ['US1','US2','US3','US4','US5','US6','US7','US8','US9',
           'US10','US11','US12','US13','US14','US15','US16','US17',
           'US18','US19','US20','US21','US22','US23','US24','Class']

df = pd.read_csv('SensorReadings24.txt', names=headers)



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

df['Class'] = df['Class'].replace(['Slight-Right-Turn','Sharp-Right-Turn','Move-Forward','Slight-Left-Turn'],['1','2','3','4'])

print(df)

#split data into train and test set 


data = np.array(df)

m,n = data.shape

#np.random.shuffle(data)

data_test = data[0:1000].T
Y_test = data_test[24]
X_test = data_test[0:n-1]

print(X_test.dtype)
data_train = data[1000:m].T
Y_train = data_train[24]
X_train = data_train[0:n-1]

Y_train = one_Hot(Y_train)
Y_test = one_Hot(Y_test)
print(Y_train)
print(Y_test)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
print(X_train.dtype)

print(Y_test.shape)
print(Y_train.shape)
print(X_test.shape)
print(X_train.shape)


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 10000)

#get the predicted test values
Predictions = make_predictions(X_test,W1,b1,W2,b2)
print(Predictions)
print(Predictions.shape)

#get the actual test values

print(f'{get_accuracy(Predictions, Y_test):.3%}')




