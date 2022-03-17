import sys
import os
from math import log, fabs
from NNetCalc import nParameters, activation_funcs
from NNetCalc import runsim as objfunc
from NNetCalc import exactlist, fusedstruct, atomcount,testexactlist, testfusedstruct, testatomcount, RCUT, maxsize

from time import time
import tensorflow as tf
import functions
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from FF_Model import ForcefieldModel


#========================================================
def lossfunc(y_predict, y_target):
    if np.any(tf.math.is_nan(y_predict)):
        return 1e8
    err = y_predict - y_target
    errmax = tf.math.reduce_max(err)
    err = tf.math.subtract(err, errmax)
    score = tf.math.square(err)
    score = tf.math.reduce_mean(score)
#    score = score.numpy()
    return score
#========================================================
def rmse(y_predict, y_target):
    err = tf.math.subtract(y_predict, y_target)
    err = tf.math.square(err)
    err = tf.math.reduce_mean(err)
    score = tf.math.sqrt(err)
    return score
#========================================================
def mae(y_predict, y_target):
    err = y_predict - y_target
    err = tf.math.abs(err)
    score = tf.math.reduce_mean(err)
    return score

#========================================================


model = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)
curweight = model.get_npweights()
cnt = 0
for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        cnt += 1
nPar = cnt

print(nPar)

if nPar != nParameters:
    raise ValueError("Model does not match expected number of Parameters")

try:
    filename = sys.argv[1]
    parameters = []
    with open(filename, 'r') as parfile:
        for line in parfile:
            newpar = float(line.split()[0])
            parameters.append(newpar)
except:
    print("Starting from Random")
    parameters = list(np.random.uniform(low=-1.5, high=1.5, size=nPar+1))


print(len(parameters))

curweight = model.get_npweights()
cnt = -1
for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        cnt += 1
        curweight[i][j] = parameters[cnt]
model.set_npweights(curweight)


#Sample training loop pulled from the Tensorflow webpage.
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
#optimizer = keras.optimizers.SGD(learning_rate=1.24e-5)

# Instantiate a loss function.
score = objfunc(parameters, usemask=False)
print("Initial Score: %s"%(score))
# Prepare the training dataset.
epochs = 2000000



for epoch in range(epochs):
    # Iterate over the batches of the dataset.
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    trainweights = model.get_weights()
    timestart = time()
    with tf.device('/CPU:0'):
        with tf.GradientTape() as tape:
            tape.watch(trainweights)
            outpair, outdens, outemb, result  = model(fusedstruct, atomcount)
            loss_value = mae(result, exactlist)
        grads = tape.gradient(loss_value, trainweights)
        for i, layer in enumerate(zip(grads, trainweights)):
            glayer, wlayer = layer
            glayer = tf.where(trainweights == 0.0, 0.0, glayer)
            grads[i] = tf.where(tf.math.is_nan(glayer), 0.0, glayer)
        optimizer.apply_gradients(zip(grads, trainweights))
#    print(trainweights)

    timeend = time()
    print("Training loss at step %d: %.4f, time:%s"
                % (epoch, float(loss_value),timeend-timestart)
            )


    if epoch % 10 == 0:
        curweight = model.get_npweights()
        cnt = -1
        oldparameters = parameters
        parameters = []
        delta = 0.0
        for i, row in enumerate(curweight):
            for j, x in np.ndenumerate(row):
                parameters.append(curweight[i][j])
#        offset = model.get_weights()[-1].numpy()
        score = objfunc(parameters, usemask=False)
#        delta = [fabs(x-y) for x,y in zip(oldparameters, parameters)]
#        maxdelta = max(delta)
#        delta = sum(delta)/float(len(delta))
#        print(
#                "Training loss at step %d: %.4f, offset: %s ,  delta: %.4E (max %.4E)"
#            % (epoch, float(loss_value), offset, delta, maxdelta)
#        )
        print("Training loss at step %d: %.4f, offset: %s"% (epoch, float(loss_value), 0.0))


curweight = model.get_npweights()
cnt = -1
endparameters = []
for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        endparameters.append(curweight[i][j])


score = objfunc(endparameters, verbose=True, usemask=False)
