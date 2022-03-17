import sys
import os
from math import log, fabs
import pickle
from NNetCalc import nParameters, prepare_input_data, activation_funcs
from NNetCalc import runsim as objfunc
import tensorflow as tf
import functions
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from FF_EAM_Fusedcall_Model import ForcefieldModel
DATA_LOCATION = "Al_EAM_processed.dat"
TEST_DATA_LOCATION = "Al_test_EAM_processed2.dat"


del os.environ['CUDA_VISIBLE_DEVICES']
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
positionlist, neighlists, distlists, atomwisedistlist, exactlist = pickle.load( open( DATA_LOCATION, "rb" ) )
test_positionlist, test_neighlists, test_distlists, test_atomwisedistlist, test_exactlist = pickle.load( open( TEST_DATA_LOCATION, "rb" ) )

X_train, X_test = prepare_input_data(positionlist, distlists, atomwisedistlist,
                       test_positionlist, test_distlists, test_atomwisedistlist)
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


RCUT = 18.0

print(len(activation_funcs))

offset = 0.0
model = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, offset=offset)
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
optimizer = keras.optimizers.Adam(learning_rate=5e-4)
#optimizer = keras.optimizers.SGD(learning_rate=1.24e-5)

# Instantiate a loss function.
score = objfunc(parameters)
print("Initial Score: %s"%(score))
# Prepare the training dataset.
epochs = 2000000
for epoch in range(epochs):
    # Iterate over the batches of the dataset.
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    trainweights = model.get_weights()
#    for layer in trainweights:
#        print(layer)
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        trainresult = model(X_train[0],X_train[1],X_train[2],X_train[3])
#        print(len(trainresult))
#        loss_value = lossfunc(trainresult, exactlist)
        loss_value = rmse(trainresult, exactlist)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
#    grads = tape.gradient(loss_value, model.model.trainable_variables)
    grads = tape.gradient(loss_value, trainweights)
    for layer in grads:
        print(layer)

    quit()


    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, trainweights))
    curweight = model.get_npweights()

#    newweight = model.get_npweights()
#    for i, row in enumerate(curweight):
#        for j, x in np.ndenumerate(row):
#            print(i,j,x,newweight[i][j])
#    quit()
    cnt = -1
    oldparameters = parameters
    parameters = []
    delta = 0.0
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            parameters.append(curweight[i][j])
    # Log every 200 batches.
    if epoch % 5 == 0:
        score = objfunc(parameters)
        delta = [fabs(x-y) for x,y in zip(oldparameters, parameters)]
        maxdelta = max(delta)
        delta = sum(delta)/float(len(delta))
        print(
                "Training loss (for one batch) at step %d: %.4f, delta: %.4E (max %.4E)"
            % (epoch, float(loss_value), delta, maxdelta)
        )


curweight = model.get_npweights()
cnt = -1
endparameters = []
for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        endparameters.append(curweight[i][j])


score = objfunc(endparameters, verbose=True)
