import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
#hyperparameters
epochs = 40
batch_size = 10
learning_rate = 0.22

def sigmoid(x):
    """
    calculating the activation
    """
    return(1/(1 + np.exp(-x)))


def input_derivative(x):
    return (1- sigmoid(x))*sigmoid(x)

class neuralnet(object):

    """
    All the important functions for the neural net can be found here.
    """

    def __init__(self,size):

        """
        Initializing the weights and the biases
        """
        self.num_layers = len(size)
        self.weights = [np.random.randn(x,y) for x,y in zip(size[1:],size[:-1])]
        self.biases =[np.random.randn(y,1) for y in size[1:]]



    def forwardpass(self,x):
        """
        Forward pass through the network. Calculate the input at each layer by multiplying the activation from the previous layer and the weights. Add bias.
        Find the output at each layer by applying sigmoid activation to the input.
        """
        in_ = []
        act =[]
        in_.append(x)
        act.append(x)
        for w,b in zip(self.weights,self.biases):
            x = np.dot(w,x) + b
            in_.append(x)
            x = sigmoid(x)
            act.append(x)
        return in_,act



    def backpropagation(self,in_,act,onehot):
        """
            This function calculates the error at each layer.
        """
        cost_deriv = act[-1] - onehot
        loss = np.sum(cost_deriv,axis = 0)
        loss = np.mean(loss)
        err_last = cost_deriv * input_derivative(in_[-1])
        error =[]
        error.append(err_last)
        for l in range(1,self.num_layers-1):
            err = np.dot(self.weights[-l].T,error[l - 1]) * input_derivative(-(l+1))
            error.append(err)
        return list(reversed(error)),loss

    def update_parameters(self,errors,act_,learning_rate):
        """
        Using the error calculated, this function updates the weights and the biases so as to reduce the loss
        """
        dw = [np.dot(error,act.T) for act,error in zip(act_[:-1],errors)]
        db =[]
        for i in range(0,self.num_layers-1):
           temp = np.sum(np.asarray(errors[i]),axis = 1)
           temp = np.reshape(temp,np.shape(self.biases[i]))
           db.append(temp)
        self.weights = self.weights - learning_rate*np.asarray(dw)
        self.biases = self.biases - learning_rate*np.asarray(db)

    def predict(self,x):
        """
            After training, this function can be used to predict
        """
        for w,b in zip(self.weights,self.biases):
            x = sigmoid(np.dot(w,x) + b)
        prediction = np.argmax(x,axis = 0)
        return prediction

df = pd.read_csv('mnist_train.csv',header = None)
data = np.asarray(df.iloc[1:].values)
n = np.shape(data)[0]
mynet = neuralnet([784,32,10])

for j in range(0,epochs):

    loss_epoch = 0
    batches =[data[k:k+batch_size] for k in range (0,n,batch_size)]
    for batch in batches:
        x = batch[:,1:].T
        x = x/255
        label = batch[:,0]
        label = [int(i) for i in label]
        one_hot = np.eye(batch_size)[label]
        in_,act = mynet.forwardpass(x)
        errors,loss= mynet.backpropagation(in_,act,one_hot)
        loss_epoch += loss
        mynet.update_parameters(errors,act,learning_rate)
    print(abs(loss_epoch))

#random testing. Can use test data too to check the neural network
x  = data[400:560,1:]
x = x/255
label = data[400:560,0]
print(label)
print(np.sum(mynet.predict(x.T)==label))
