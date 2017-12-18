'''
Author: G Siva Perumal

This code does the following:
1. process the CIFAR data downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html into numpy arrays
2. Build a neural network to detect objects from the 20 classes in CIFAR dataset

'''


import tensorflow as tf
import numpy as np
import os
import argparse 
import tarfile
import urllib
import pickle
import random
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator

#hyperparameters
no_epochs = 250
batch_size = 128
no_batches = int(50000/128)

parser = argparse.ArgumentParser(description='Neural network argument')
parser.add_argument('--data_down', default='True', help='to download the data or not')
args = parser.parse_args()

#path to the data folder
path = os.path.join(os.getcwd())

#path to the test data
test_path = os.path.join(path,'CIFAR/cifar-10-batches-py/test_batch')


def read_file(file):
    '''read the data from the path specified'''
    with open(file,'rb') as data_file:
        data = pickle.load(data_file,encoding = 'bytes')
        x,y = data[b'data'],data[b'labels']
        data_file.close()
    return x,y
    
def one_hot_convertor(y):
    '''converts the input vector into a one hot vector'''
    Onehot_operation = OneHotEncoder(sparse=False)
    Onehot = Onehot_operation.fit_transform(y)
    Onehot =Onehot.astype('float32')
    return Onehot
    

def get_data(batchno):
    ''' Randomly shuffles the batch specified by batchno '''
    start = batch_size * batchno
    end = batch_size * batchno + batch_size
    random_index = list(range(start,end))
    random.shuffle(random_index)
    x_batch = x_train[random_index,:,:,:]
    y_batch = Onehot_train[random_index]
    return x_batch,y_batch


#Download the dataset if needed
if(args.data_down):

    path = os.getcwd()
    dataset_name = "CIFAR"
    if not os.path.exists(path + '/' + dataset_name):
        os.makedirs(path + '/' + dataset_name)
    compressed_file = urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', path + '/' + dataset_name + '/cifar.tgz' )
    cifar_tgz = tarfile.open (os.path.join(path,dataset_name,'cifar.tgz'))
    cifar_tgz.extractall(os.path.join(path,dataset_name))
    cifar_tgz.close()
    
#shape of the training data and labels
x_train = np.zeros((50000,3072))
y_train = np.zeros((50000))

#read the data from the files and store them in x_train and y_train
files = ['data_batch_1', 'data_batch_2','data_batch_3','data_batch_4','data_batch_5']
for count,file in enumerate(files):
    new_path = os.path.join(path,'CIFAR/cifar-10-batches-py',file)
    x_train[count*10000:(count+1)*10000,:],y_train[count*10000:(count+1)*10000] = read_file(new_path)  
x_train = x_train.reshape(50000,3,32,32)
x_train = x_train.transpose((0, 2, 3, 1))
y_train = y_train.reshape((y_train.shape[0],1))

#find parameters to normalize the data
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))

#normalize the train data
x_train = (x_train-mean)/(std+1e-7)

#Read the test data and normalize 
x_test,y_test = read_file(test_path)
x_test = x_test.reshape((10000,3,32,32))
x_test = x_test.transpose((0,2,3,1))
x_test = (x_test-mean)/(std+1e-7)
y_test = np.array(y_test)
y_test = y_test.reshape((y_test.shape[0],1))

#Convert the train and test labels into one hot_vectors
Onehot_train = one_hot_convertor(y_train)
Onehot_test = one_hot_convertor(y_test)


#data augmentation for improved training
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)


#Create tensorflow graph
X = tf.placeholder(tf.float32,(None,32,32,3))
Y = tf.placeholder(tf.float32,(None,10))
training = tf.placeholder(tf.bool)

#architecture

#2 convolutional layers with batch normalization followed by a maxpooling layer
cl1 = tf.layers.conv2d(X,48,3,padding = 'same')
bn1 = tf.layers.batch_normalization(cl1, training = training)
bn1 = tf.nn.relu(bn1)

cl2 = tf.layers.conv2d(bn1,48,3,padding = 'same')
bn2 = tf.layers.batch_normalization(cl2, training = training)
bn2 = tf.nn.relu(bn2)

mp1 = tf.layers.max_pooling2d(bn2,2,2,padding = 'same')
mp1 = tf.layers.dropout(mp1,rate = 0.25,training = training)

#2 convolutional layers with batch normalization followed by a maxpooling layer
cl3 = tf.layers.conv2d(mp1,96,3,padding = 'same')
bn3 = tf.layers.batch_normalization(cl3, training = training)
bn3 = tf.nn.relu(bn3)

cl4 = tf.layers.conv2d(bn3,96,3, padding = 'same')
bn4 = tf.layers.batch_normalization(cl4, training = training)
bn4 = tf.nn.relu(bn4)

mp2 = tf.layers.max_pooling2d(bn4,2,2)
mp2 = tf.layers.dropout(mp2,rate = 0.25,training = training)

#2 convolutional layers with batch normalization followed by a maxpooling layer
cl5 = tf.layers.conv2d(mp2,196,3,padding = 'same')
bn5 = tf.layers.batch_normalization(cl5, training = training)
bn5 = tf.nn.relu(bn5)

cl6 = tf.layers.conv2d(bn5,196,3,padding = 'same')
bn6 = tf.layers.batch_normalization(cl6, training = training)
bn6 = tf.nn.relu(bn6)

mp3 = tf.layers.max_pooling2d(bn6,2,2)

mp3_shape = mp3.shape.as_list()
fc1 = tf.reshape(mp3,(-1,mp3_shape[1] *mp3_shape[2]* mp3_shape[3]))
fc2 = tf.layers.dense(fc1,512,activation = tf.nn.relu)
fc2 = tf.layers.dropout(fc2,rate = 0.5,training = training)
fc3 = tf.layers.dense(fc2,256,activation = tf.nn.relu)
fc3 = tf.layers.dropout(fc3,rate = 0.5,training = training)
fc4 = tf.layers.dense(fc3,10, activation = tf.nn.softmax)


#calculating predictions
pred_prob = tf.nn.softmax(fc4)
pred = tf.argmax(pred_prob,axis = 1)
ground_truth = tf.argmax(Y,axis = 1)

#calculating accuracy and loss 
correct_predictions = tf.cast(tf.equal(pred,ground_truth),tf.float32)
acc = tf.reduce_mean(correct_predictions)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc4,labels = Y))

#Defining the optimizer and the training_operation
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0001,decay=1e-6)
	training_operation = optimizer.minimize(loss)

#Initialization operation for the variables of the network
init = tf.global_variables_initializer()


#Create a session to run the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    for epochs in range(1,no_epochs + 1):
        batch = 0
        for x_batch,y_batch in datagen.flow(x_train,Onehot_train, batch_size = batch_size):  
            batch = batch + 1          
            sess.run([training_operation],feed_dict = {X:x_batch,Y:y_batch,training: True})
            if (batch > no_batches):
                break

        l,a = sess.run([loss,acc],feed_dict = {X:x_test, Y:Onehot_test,training: False})
        print("The loss after epoch{} is {} and the accuracy is {}".format(epochs,l,a))
    
    print("Optimization done! Ready to predict")
    a_test = acc.eval(feed_dict = {X:x_test,Y:Onehot_test,training:False})
    print('###########The final acc is {}'.format(a_test))
 

