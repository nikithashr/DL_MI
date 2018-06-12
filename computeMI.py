#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:29:05 2018

@author: NikithaShravan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 00:48:56 2018

@author: NikithaShravan
"""

import keras 
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np

params = {}
params['layers'] = [1024, 20, 20, 20]
params['batch_size'] = 128
params['num_epochs'] = 1000
params['learning_rate'] = 1e-2
params['activation'] = 'relu'
params['optimizer'] = 'sgd'
arch_name =  '-'.join(map(str,params['layers']))
params['save_dir'] = 'rawdata/' + params['activation'] + '_' + arch_name
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
                          
    return X_train[:500], Y_train[:500], y_train[:500], X_test[:100], Y_test[:100], y_test[:100]


def build_network(activation, layer_dims, input_dims, num_classes):
    """ 
    layer_dims: excluding input and output dimensions. 
    """
    model = Sequential()
    
    model.add(Dense(layer_dims[0],input_dim=input_dims))   
    #model.add(Dropout())
    
    for i in range(1,len(layer_dims)):
        model.add(Activation(activation))
        model.add(Dense(layer_dims[i]))
        #model.add(Dropout())
    
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
    return model
   
X_train, Y_train, y_train, X_test, Y_test, y_test = load_data()
print("SHAPES: ")
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
#data = np.random.random( size = (1000, 20) )
num_classes = 10 
#y = np.random.randint(0, num_classes, 1000)
#X_train = data
#Y_train = keras.utils.to_categorical(y, num_classes)
params['input_dim'] = X_train.shape[1]
params['output_dim'] = Y_train.shape[1]
model = build_network(params['activation'], params['layers'], params['input_dim'], params['output_dim'])


trained_epochs = []
output_epochs = []
from keras.callbacks import LambdaCallback
tracker_cb = LambdaCallback(on_epoch_begin=lambda epoch, logs: output_epochs.append([layerfuncs([X_train,])[0] for layerfuncs in [K.function(model.inputs, [l.output,]) for l in model.layers if epoch<20 or epoch%100 == 0]]))
r = model.fit(x=X_train, y=Y_train, 
              verbose    = 2, 
              batch_size = params['batch_size'],
              epochs     = params['num_epochs'],
#              validation_data=(X_test, Y_test),
              callbacks=[tracker_cb])

""" Saving activations at every layer """
epoch = 1
import os
if not os.path.exists(params['save_dir']):
    print("Making directory ", params['save_dir'])
    os.makedirs(params['save_dir'])
    
inp = model.input
for endx, epochout in enumerate(output_epochs):
#    functors = [K.function([inp], [out]) for out in each_epoch_out]
#    layer_outputs = [func([X_train, 0.])[0] for func in functors]
#    l = {}
##    num_layers = 1
##    layerfuncs[lndx]([X_train,])[0]
#    print("output epochs shape: ", np.shape(output_epochs))
#    for lndx, layerfuncs in enumerate(epochfuncs):
#        l[lndx] = layerfuncs([X_train,])[0]
##        print("shape of each layer output:", l.shape)
##        num_layers += 1
    fname = params['save_dir'] + "/epoch%03d"% epoch
    with open(fname, 'wb') as f:
        import _pickle as cPickle
        cPickle.dump({'activation':params['activation'], 'layer_outputs': epochout}, f)
    epoch += 1
#    inp = layer_outputs
    


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


#data = np.random.random( size = (1000, 20) ) 
#y = np.random.randint(0,10,1000)



#print(H_M_lb)
#print(H_M_ub)
#print(H_M_given_X)
#print(H_M_given_Y_lb)
#print(H_M_given_Y_ub)


Epoch_MI_lb = {}
Epoch_MI_ub = {}
nats2bits = 1.0/np.log(2)
for i in range(1,params['num_epochs']+1):
    fname = params['save_dir'] + "/epoch%03d"% i
    print(fname)
    with open(fname, 'rb') as f:
        d = cPickle.load(f)
#    print("epoch number: ", i)
    print("BEGINNING EPOCH")
    itr = 1
    layer_MI_lb = []
    layer_MI_ub = []

    for layer_out in d['layer_outputs']:
#        print("layer out shape: ", layer_out.shape)
        itr += 1
        if itr % 2 == 1:
            continue
        H_M_given_X = kde_condentropy(layer_out, noise_variance) 
        H_M_ub = entropy_func_upper([layer_out,])[0]
        H_M_lb = entropy_func_lower([layer_out,])[0]
        H_M_given_Y_ub, H_M_given_Y_lb = y_condentropy(layer_out, y_train, 10, noise_variance)
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
#        
        layer_MI_lb.append([MI_X_M_lb, MI_Y_M_lb]) 
        layer_MI_ub.append([MI_X_M_ub, MI_Y_M_ub])
    Epoch_MI_lb[i]=layer_MI_lb
    Epoch_MI_ub[i]=layer_MI_ub
    


#%matplotlib inline
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import seaborn as sns
#sns.set_style('darkgrid')    
max_epoch, COLORBAR_MAX_EPOCHS = params['num_epochs'], params['num_epochs'] 
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []

#fig=plt.figure(figsize=(10,5))
#for actndx, (activation, vals) in enumerate(Epoch_MI_ub.items()):
#
#    plt.subplot(1,2,actndx+1)    
#    for epoch in epochs:
#        c = sm.to_rgba(epoch)
#        xmvals = np.array(vals[epoch]['MI_XM_'+infoplane_measure])[PLOT_LAYERS]
#        ymvals = np.array(vals[epoch]['MI_YM_'+infoplane_measure])[PLOT_LAYERS]
#
#        plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
#        plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)
#    
#    plt.ylim([0, 3.5])
#    plt.xlim([0, 14])
#    plt.xlabel('I(X;M)')
#    plt.ylabel('I(Y;M)')
#    plt.title(activation)
#    
#cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
#plt.colorbar(sm, label='Epoch', cax=cbaxes)
#plt.tight_layout()


fig=plt.figure(figsize=(10,5))
#plt.subplot(1,2,1)
for epoch, layer_ub in enumerate(Epoch_MI_ub.items()):
#    epochs = sorted(vals.keys())
#    if not len(epochs):
#        continue
    layer_ub = np.array(layer_ub[1])    
#    for epoch in epochs:
    c = sm.to_rgba(epoch)
    xmvals = np.array(layer_ub[:,0])
    ymvals = np.array(layer_ub[:,1])
#    print("X vals: ", xmvals)
#    print("Y vals: ", ymvals)

    plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
    plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in range(layer_ub.shape[0])], edgecolor='none', zorder=2)


#    plt.show()
#    plt.hold(True)
#    plt.title(activation)
plt.ylim([0, 3.5])
plt.xlim([0, 14])
plt.xlabel('I(X;M)')
plt.ylabel('I(Y;M)')    
#cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
#plt.colorbar(sm, label='Epoch', cax=)
plt.tight_layout()
#plt.show()


#if DO_SAVE:
plt.savefig('plots_' +params['activation'] + '_' + arch_name + '.png')
        
        
        
    
 

