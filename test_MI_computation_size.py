#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:05:45 2018

@author: NikithaShravan
"""

import keras 
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np

def load_data():
    
    num_classes = 10

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)
    
    num_rows = X_train.shape[1]
    num_cols = X_train.shape[2]
    #num_channels = X_train.shape[3]
    input_dims = num_rows*num_cols 
#    print(X_train.shape)
    
    X_train = X_train.reshape(X_train.shape[0], input_dims)
    X_test = X_test.reshape(X_test.shape[0], input_dims)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
                          
    return X_train, Y_train, y_train, X_test, Y_test, y_test


""" Entropy: H(M|X)  """
def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1) 
   
""" Get Pairwise distances with Keras Backend """
def get_dists_backend(X):
    norms = K.expand_dims(K.sum(K.square(X), axis=1),1)
    return norms + K.transpose(norms) - 2*K.dot(X,K.transpose(X))

""" Entropy: H(M) upper bound """
def entropy_upper(data, noise_variance):
    pairwise_dists = get_dists_backend(data)
    pairwise_dists /= (2*noise_variance)
    
    N = K.cast(K.shape(data)[0], K.floatx())
    dims = K.cast(K.shape(data)[1], K.floatx())
    normconst = (dims/2.0)*K.log(2*np.pi*noise_variance)
    
    term1 = K.logsumexp(-pairwise_dists, axis=1) - K.log(N) - normconst
    
    return -K.mean(term1) + dims/2

""" Entropy: H(M) lower bound """
def entropy_lower(data, noise_variance):
    h_upper = entropy_upper(data,4*noise_variance)
    dims = K.cast(K.shape(data)[1], K.floatx())
    
    return h_upper + np.log(0.25)*dims/2

""" Entropy: H(M|Y) """
def y_condentropy(data, y, num_classes, noise_variance):
    # extract all the data points of a particular class,
    get_label_ind = {}
    for i in range(num_classes):
        get_label_ind[i] = y==i #y in Nx1 array of classes
    # get the probability of occurence of each class
    probs = np.mean(keras.utils.to_categorical(y, num_classes),axis=0)
    # get their entropies
    
    Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder 
    entropy_func_upper = K.function([Klayer_activity,], [entropy_upper(Klayer_activity, noise_variance),])
    entropy_func_lower = K.function([Klayer_activity,], [entropy_lower(Klayer_activity, noise_variance),])

    sum_entropy_lb, sum_entropy_ub = 0,0
    for i in range(num_classes):
        h_upper = entropy_func_upper([data[get_label_ind[i],:],])[0]
        h_lower = entropy_func_lower([data[get_label_ind[i],:],])[0]
        sum_entropy_lb += probs[i]*h_lower
        sum_entropy_ub += probs[i]*h_upper
    return sum_entropy_ub, sum_entropy_lb
# sum the above products

    
    
""" Computing Mutual Information """
noise_variance = 1e-1
Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder 
entropy_func_upper = K.function([Klayer_activity,], [entropy_upper(Klayer_activity, noise_variance),])
entropy_func_lower = K.function([Klayer_activity,], [entropy_lower(Klayer_activity, noise_variance),])


""" Test I(X;Y) on MNIST Data """
X_train, Y_train, y_train, X_test, Y_test, y_test = load_data()
#X_train = X_train[:500]
#y_train = y_train[:500]
print("SHAPES: ")
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
#data = np.random.random( size = (1000, 20) )
num_classes = 10 


H_M_given_X = kde_condentropy(X_train, noise_variance) 
H_M_ub = entropy_func_upper([X_train,])[0]
H_M_lb = entropy_func_lower([X_train,])[0]
H_M_given_Y_ub, H_M_given_Y_lb = y_condentropy(X_train, y_train, 10, noise_variance)
MI_X_M_lb = (H_M_lb - H_M_given_X)/np.log(2)
MI_X_M_ub = (H_M_ub - H_M_given_X)/np.log(2)
MI_Y_M_lb = (H_M_lb - H_M_given_Y_lb)/np.log(2)
MI_Y_M_ub = (H_M_ub - H_M_given_Y_ub)/np.log(2)
print(":" )
print("MI(X;M) lower bound:  " , MI_X_M_lb)
print("MI(X;M) upper bound:  " , MI_X_M_ub)
print(":" )
print("MI(M;Y) lower bound:  " , MI_Y_M_lb)
print("MI(M;Y) upper bound:  " , MI_Y_M_ub)
