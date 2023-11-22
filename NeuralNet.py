''''
Class to implement structure and methods for ANN

'''

import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork():

    def __init__(self, layers=[24,10,4], learning_rate=0.15, iterations=10000):

        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.layers = layers
        self.error = []
        self.iters = []
        self.X = None
        self.y = None

    def init_weights(self):
        '''Randomly initilise the weights'''

        self.params['W1'] = np.random.rand(self.layers[1], self.layers[0])
        self.params['b1'] = np.random.rand(self.layers[1], 1)
        self.params['W2'] = np.random.rand(self.layers[2], self.layers[1])
        self.params['b2'] = np.random.rand(self.layers[2], 1)

    def relu(self, Z):
        '''ReLU Activation Function'''
        return np.maximum(Z,0)
    
    def dRelu(self, Z):
        '''Derivative of the ReLU function'''
        return Z>0
    
    def softmax(self, Z):
        '''Softmax activation function'''
        exp = np.exp(Z-np.max(Z))
        return exp / exp.sum(axis=0)
    
    def forward_propogation(self):
        '''Performs the forward propogation'''

        Z1 = self.params['W1'].dot(self.X) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = self.params['W2'].dot(A1) + self.params['b2']
        A2 = self.softmax(Z2)

        #save the parameters
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return A2

    def back_propogation(self, A2, m):
        '''Perfroms the backwards propogation'''

        dZ2 = 2*(A2 - self.y)
        dW2 = 1/m * (dZ2.dot(self.params['A1'].T))
        db2 = 1/m * np.sum(dZ2,1)
        dZ1 = self.params['W2'].T.dot(dZ2)*self.dRelu(self.params['Z1'])
        dW1 = 1/m * (dZ1.dot(self.X.T))
        db1 = 1/m * np.sum(dZ1,1)

        #save the parameters
        self.params['dW1'] = dW1
        self.params['db1'] = db1
        self.params['dW2'] = dW2
        self.params['db2'] = db2

    def update_params(self):
        '''update the weights and bias' of the network'''

        self.params['W1'] = self.params['W1'] - self.learning_rate * self.params['dW1']
        self.params['b1'] = self.params['b1'] - self.learning_rate * np.reshape(self.params['db1'], (10,1))
        self.params['W2'] = self.params['W2'] - self.learning_rate * self.params['dW2']
        self.params['b2'] = self.params['b2'] - self.learning_rate * np.reshape(self.params['db2'], (4,1))

    def get_prediction(self,A2):
        '''Get the models predictions'''
        return np.argmax(A2, axis=0)

    def get_accuracy(self, predictions, y):
        '''Get the accuracy of the network'''
        Y = np.argmax(y, axis=0)
        return np.sum(predictions == Y)/Y.size

    def mean_sqr_error(self, predictions, y):
        '''calculate the MSE of the network'''
        return np.square(np.subtract(y, predictions)).mean()
    
    def gradient_descent(self, X, y):
        '''Trains the NN using the supplied data & labels'''

        self.X = X
        self.y = y
        self.init_weights()

        s, m = X.shape

        for i in range(self.iterations):
            A2 = self.forward_propogation()
            self.back_propogation(A2,m)
            self.update_params()

            if(i+1) % int(self.iterations/10) ==0:
                print(f"Iterations: {i+1} / {self.iterations}")
                prediction = self.get_prediction(A2)
                print(f'{self.get_accuracy(prediction, y):.3%}')

                mse = self.mean_sqr_error(A2, y)
                print(f"MSE::{mse}")

                self.error.append(mse)
                self.iters.append(i+1)

        self.plot_error()

        return self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2']

    def plot_error(self):
        '''plot the error MSE vs Epoch''' 

        plt.scatter(self.iters, self.error)
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('MSE vs Epoch')
        plt.show()

    def make_prediction(self,X, y):
        '''make a prediction using the trained network'''      

        Z1 = self.params['W1'].dot(X) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = self.params['W2'].dot(A1) + self.params['b2']
        A2 = self.softmax(Z2)

        mse = self.mean_sqr_error(A2, y)
        prediction = self.get_prediction(A2)
        
        return prediction, mse        

    