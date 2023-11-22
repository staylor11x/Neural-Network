'''
    Class to implement the structure for the neural network
'''

import numpy as np
from matplotlib import pyplot as plt



class NueralNet():

    '''Three-layer neural network'''
    
    def __init__(self, layers=[24,10,4], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.x = None
        self.y = None

    def init_weights(self):
        '''Initlise the weights for a normal random distribution'''

        np.random.seed(1)       #seed for the random number

        self.params['W1'] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.rand(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.rand(self.layers[2],)

    def relu(self, Z):
        '''Relu activation function - performs a thershold operation to each 
            input element where values less than zero are set to zero.
        '''
        return np.maximum(0,Z)
    
    def dRelu(self, x):
        '''the derivative of the relu function'''

        x[x<=0] = 0
        x[x>0] = 1
        return x     
    
    def sigmoid(self, Z):
        '''
            The sigmoid function takes a real number in any range
            and squashes it to a real-valued output between 0 and 1
        '''

        return 1/(1+np.exp(-Z))
    
    def eta(self, x):
        ETA = 0.0000000001
        return np.maximum(x,ETA)
    
    def entropy_loss(self,y,yhat):
        '''This measures how good the networks predictions are'''

        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat)
        yhat_inv = self.eta(yhat_inv)
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat),y) + np.multiply((y_inv),np.log(yhat_inv))))

        return loss

    def forward_propogation(self):
        '''performs forward propogation'''

        Z1 = self.X.dot(self.params['W1']) * self.params['b1']              #compute the weighted sum between the input and the first layer weights, then add the bias
        A1 = self.relu(Z1)                                                  #pass result throught the relu function 
        Z2 = A1.dot(self.params['W2']) + self.params['b2']                  #compute the weightted sum between the input and the second layer weights, then add the bias
        yhat = self.sigmoid(Z2)                                             #pass result throught the sigmoid function
        loss = self.entropy_loss(self.y, yhat)                              #evaliate the effectiveness of the network

        #save the caluclated parameters (these will be used later in back propogation)

        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat, loss
    
    def back_propogation(self, yhat):
        '''
            Computes the derivates and updates the weights and bias accordingly

            --  calculate the derivates of the Relu
            --  Then calculate and save the derivative of every parameter wrt the loss function
        '''

        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.y, self.eta(yhat))
        d1_wrt_sig  = yhat * (yhat_inv)
        dl_wrt_z2   = dl_wrt_yhat * d1_wrt_sig

        dl_wrt_A1   = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2   = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2   = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1   = dl_wrt_A1 * self.dRelu(self.params['Z1'])
        dl_wrt_w1   = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1   = np.sun(dl_wrt_z1, axis=0, keepdims=True)


        #update the weights and bias
        #subtract the derivative multiplied by the learning rate, the learning rate tells the network how big the update should be 
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2


    def fit(self, X, y):
        '''
            Trains the NN using the specified data and labels

            X = input dataset
            y = labels
        '''

        self.X = X
        self.y = y

        self.init_weights() #initilise the weights and bias

        #call the forward and back propogation methods for a specified number of itterations (Learning)
        for i in range(self.iterations):
            yhat, loss = self.forward_propogation()
            self.back_propogation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        '''make predictions on the test data'''

        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)

        return np.round(pred)
    
    def acc(self, y, yhat):
        '''calculates the accuracy between the predicted values and the truth labels'''

        acc = int(sum(y == yhat)/len(y) *100)
        return acc
    
    def plot_loss(self):
        '''plot the loss curve'''

        plt.plot(self.loss)
        plt.xlabel("itterations")
        plt.ylabel("Loss curve for training")
        plt.show()