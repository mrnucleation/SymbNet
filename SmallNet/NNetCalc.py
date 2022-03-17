import os
import natsort
from glob import glob
from random import random, shuffle
from time import time
from vaspUtilities import getTotalEnergy, getForces, getPositions, getPosForces
from EAMCalc import eamenergies

import numpy as np
import warnings
import functions
from math import fabs
from FF_Model import ForcefieldModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

activation_funcs = {
    'pair':
    [[
      *[functions.Constant()] * 2,
      *[functions.Identity()] * 2,
#      *[functions.Square()] * 1,
#      *[functions.NegExp()] * 1,
      *[functions.Pow(power=-12.0)] * 1,
      *[functions.Pow(power=-9.0)] * 1,
      *[functions.Pow(power=-6.0)] * 1,
      *[functions.Pow(power=-3.0)] * 1,
      *[functions.Pow(power=-1.0)] * 1,
    ],
    [
      *[functions.Constant()] * 2,
      *[functions.Identity()] * 2,
#      *[functions.Square()] * 1,
#      *[functions.Sin()] * 2,
#      *[functions.Cos()] * 2,      
#      *[functions.Sigmoid()] * 1,
    ]],
    
    'density':
    [[
    *[functions.Constant()] * 1,
    *[functions.Identity()] * 1,
#    *[functions.Square()] * 1,
#    *[functions.Sin()] * 1,
#    *[functions.Cos()] * 1,
#    *[functions.Pow(power=-1.0)] * 1,
    *[functions.Pow(power=-6.0)] * 1,
    *[functions.Pow(power=-3.0)] * 1,
    *[functions.Pow(power=-1.0)] * 1,
#    *[functions.Sigmoid()] * 1,        
    ]],
    
    'embedding':
    [[
#    *[functions.Constant()] * 1,
    *[functions.Identity()] * 1,
#    *[functions.Square()] * 1,
    *[functions.Sqrt()] * 1,
#    *[functions.Sin()] * 1,
#    *[functions.Cos()] * 1,
#    *[functions.Sigmoid()] * 1,
#    *[functions.Pow(power=-1.0)] * 1,
#    *[functions.Pow(power=-2.0)] * 1,
    ],
    [
#    *[functions.Constant()] * 1,
    *[functions.Identity()] * 1,    
    *[functions.Square()] * 1,
#    *[functions.Sin()] * 1,
#    *[functions.Cos()] * 1,      
    *[functions.Identity()] * 1,
#    *[functions.Sigmoid()] * 1,
    ]]
}



zonewidth = 0.35
xhigh = zonewidth/2.0
xlow = -zonewidth/2.0


print("Using a Deadzone of width: %s"%(zonewidth))
print("Deadzone low/high: %s, %s"%(xlow, xhigh))
def deadzone(par, xlo, xhi):
    if par >= xhi:
        return par-xhi, 1
    elif par <= xlo:
        return par-xlo, 1
    else:
        return 0.0, 0

def inv_deadzone(par, xlo, xhi):
    if par >= 0:
        return par+xhi
    elif par <= 0:
        return par+xlo
    else:
        return 0.0
RCUT = 132.0

workdir = os.getcwd() + "/"

if os.path.exists(workdir+"Al_TestData.npy"):
    testfusedstruct = np.load("Al_TestData.npy")
    testexactlist = np.load("Al_TestEnergies.npy")
    testatomcount = np.load("Al_TestSizes.npy")
    fusedstruct = np.load("Al_TrainData.npy")
    exactlist = np.load("Al_TrainEnergies.npy")
    atomcount = np.load("Al_TrainSizes.npy")
    ntrainstruct = exactlist.size
    nteststruct = testexactlist.size
    ntotalstruct = ntrainstruct + nteststruct
    maxsize = max(atomcount.max(), testatomcount.max())
    dummymodel = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)
else:
    structdir = os.environ['STRUCTDIR']
    elementsymb = structdir.split("/")[-2].strip()
    structlist = natsort.natsorted(glob(structdir + "*/*/*.geo"))
#    monomerCalcLoc = os.environ['MONODIR'] + "/OUTCAR"
#    monomerEnergy = float(getTotalEnergy(monomerCalcLoc))
#    print("Monomer: %s"%(monomerEnergy))

    if len(structlist) < 1:
        raise IOError("No input structures have been found!")

    maxsize = 0
#    newstruct = []
    for struct in structlist:
        size = int(struct.split('/')[-3])
        maxsize = max(size, maxsize)
#        if size < 3:
#            maxsize = max(size, maxsize)
#            newstruct.append(struct)
#    structlist = newstruct
    dummymodel = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)

    ntotalstruct = len(structlist)
    nteststruct = round(0.2*len(structlist))
    ntrainstruct = ntotalstruct-nteststruct
    shuffle(structlist)
    struct_train = structlist[:ntrainstruct]
    struct_test = structlist[ntrainstruct:]


    exactlist = []
    atomcount = []
    fusedstruct = None
    #Process the input data into
    print("Train Set Size:%s"%(ntrainstruct))
    for struct in struct_train:
        geoFile = struct.split('/')[-1]
        geoFile = os.path.dirname(struct) + "/" + geoFile
        size = int(struct.split('/')[-3])
        atomcount.append(size)
        outcar = os.path.dirname(struct) + '/OUTCAR'
        energy = eamenergies(geoFile)
#        print(size, energy)
        exactlist.append(energy)
        refPositions, refForces = getPosForces(outcar)
        refPositions = np.array(refPositions, dtype=np.float32)
        neilist = dummymodel.buildneighlist(refPositions)
        distlist = dummymodel.builddistlist(refPositions, neilist)
        if fusedstruct is None:
            fusedstruct = distlist
        else:
            fusedstruct = np.concatenate((fusedstruct,distlist), axis=0)
    exactlist = np.array(exactlist, dtype=np.float32)
    atomcount = np.array(atomcount, dtype=np.int32)
#    atomcount = np.array(atomcount, dtype=np.float32)

    testexactlist = []
    testatomcount = []
    testfusedstruct = None
    #Process the input data into
    print("Test Set Size:%s"%(nteststruct))
    for struct in struct_test:
        geoFile = struct.split('/')[-1]
        geoFile = os.path.dirname(struct) + "/" + geoFile
        size = int(struct.split('/')[-3])
        testatomcount.append(size)
        outcar = os.path.dirname(struct) + '/OUTCAR'
        energy = eamenergies(geoFile)
#        print(size, energy)
        testexactlist.append(energy)
        refPositions, refForces = getPosForces(outcar)
        refPositions = np.array(refPositions, dtype=np.float32)
        neilist = dummymodel.buildneighlist(refPositions)
        distlist = dummymodel.builddistlist(refPositions, neilist)
        if testfusedstruct is None:
            testfusedstruct = distlist
        else:
            testfusedstruct = np.concatenate((testfusedstruct,distlist), axis=0)
#        print(len(testexactlist))
    testexactlist = np.array(testexactlist, dtype=np.float32)
    testatomcount = np.array(testatomcount, dtype=np.int32)
    print(exactlist.shape)
    print(fusedstruct.shape)
    print(atomcount.shape)

    print(testexactlist.shape)
    print(testfusedstruct.shape)
    print(testatomcount.shape)
    np.save("Al_TestData", testfusedstruct)
    np.save("Al_TestEnergies", testexactlist)
    np.save("Al_TestSizes", testatomcount)
    np.save("Al_TrainData", fusedstruct)
    np.save("Al_TrainEnergies", exactlist)
    np.save("Al_TrainSizes", atomcount)

frac_train = ntrainstruct/float(ntotalstruct)
frac_test = nteststruct/float(ntotalstruct)

curweight = dummymodel.get_npweights()
nParameters = 0
for i, row in enumerate(curweight):
    nParameters += row.size
print("Number of Parameters in the Model: %s"%(nParameters))

print(exactlist.shape)
print(fusedstruct.shape)
print(atomcount.shape)

print(testexactlist.shape)
print(testfusedstruct.shape)
print(testatomcount.shape)

exactlist = tf.Variable(exactlist)
fusedstruct = tf.Variable(fusedstruct)
atomcount = tf.Variable(atomcount, dtype=tf.float32)
testexactlist = tf.Variable(testexactlist)
testfusedstruct = tf.Variable(testfusedstruct)
testatomcount = tf.Variable(testatomcount, dtype=tf.float32)


#========================================================
def nplossfunc(y_predict, y_target):
    if np.any(tf.math.is_nan(y_predict)):
        return 1e18
    err = y_predict - y_target
    errmax = tf.math.reduce_max(err)
    err = tf.math.subtract(err, errmax)
    score = tf.math.square(err)
    score = tf.math.reduce_mean(score)
    score = score.numpy()
    return score
#========================================================
def rmse(y_predict, y_target):
    if np.any(tf.math.is_nan(y_predict)):
        return 1e18
    err = tf.math.subtract(y_predict, y_target)
    err = tf.math.square(err)
    err = tf.math.reduce_mean(err)
    score = tf.math.sqrt(err)
    return score.numpy()
#========================================================
def mae(y_predict, y_target, numpyout=True):
    if np.any(tf.math.is_nan(y_predict)):
        if numpyout:
            return 1e18
        else:
            print(y_predict)
            return tf.constant(1e18)
    err = y_predict - y_target
    err = tf.math.abs(err)
    score = tf.math.reduce_mean(err)
    if numpyout:
        return score.numpy()
    else:
        return score
#========================================================
def runsim(parameters, verbose=False, usemask=True):
    starttime = time()
    model = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)
    if usemask:
        maskedweights = [deadzone(x, xlow, xhigh) for x in parameters]
        maskedweights, parmask = list(map(list, zip(*maskedweights)))
        maskcnt = sum(parmask)
    else:
        maskedweights = parameters
        maskcnt = 0

    curweight = model.get_npweights()
    cnt = -1
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            cnt += 1
            curweight[i][j] = maskedweights[cnt]
    model.set_npweights(curweight)

    if verbose:
        model.pretty_output()
    cnt = 0

        
    with warnings.catch_warnings():    
        warnings.filterwarnings('ignore', r'overflow encountered in')
        warnings.filterwarnings('ignore', r'overflow encountered in reduce')
        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
        warnings.filterwarnings('ignore', r'Input contains NaN')

        with tf.device('/CPU:0'):
            result = model(fusedstruct, atomcount)
            maescore = mae(result, exactlist)
            score = maescore
#            score = nplossfunc(result, exactlist)
            if verbose:
#                maescore = mae(result, exactlist)
                rmsescore = rmse(result, exactlist)
                print("Train MAE: %s, Train RMSE: %s"%(maescore, rmsescore))
#            score = maescore

    if verbose:
        with open("corr_train.dat", "w") as outfile:
            for e_trial, e_ref in zip(result.numpy(), exactlist.numpy()):
                outfile.write("%s %s\n"%(e_trial, e_ref))

    if np.isinf(score) or np.isnan(score):
        score = 1e18

    if score > 1e18:
        score = 1e18
        testscore = 1e18
    else:
        with tf.device('/CPU:0'):
            testresult = model(testfusedstruct, testatomcount)
            maescore = mae(testresult, testexactlist)
            testscore = maescore
#            testscore = nplossfunc(testresult, testexactlist)
            if verbose:
#                maescore = mae(testresult, testexactlist)
                rmsescore = rmse(testresult, testexactlist)
                print("Test MAE: %s, Test RMSE: %s"%(maescore, rmsescore))
#            testscore = maescore
    if np.isinf(testscore) or np.isnan(testscore):
        testscore = 1e18

    if testscore > 1e18:
        testscore = 1e18

    if verbose:
        with open("corr_test.dat", "w") as outfile:
            for e_trial, e_ref in zip(testresult.numpy(), testexactlist.numpy()):
                outfile.write("%s %s\n"%(e_trial, e_ref))

    totalscore = frac_test*testscore + frac_train*score

    endtime = time()
    tf.keras.backend.clear_session()
    print("Score: %s, Runtime: %s"%(score, endtime-starttime))
    if not verbose:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s | %s | %s  \n'%(outstr, testscore, score, totalscore))
        with open("dumpfile_mask.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in maskedweights])
            outfile.write('%s | %s | %s | %s  \n'%(outstr, testscore, score, totalscore))

    else:
        print("Test Score:%s"%(testscore))
    return score
#========================================================
def tf_minimize(parameters, nepoch = 200, usemask=True):
    starttime = time()
    model = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)
    if usemask:
        maskedweights = [deadzone(x, xlow, xhigh) for x in parameters]
        maskedweights, parmask = list(map(list, zip(*maskedweights)))
        maskcnt = sum(parmask)
    else:
        maskedweights = parameters
        maskcnt = 0
        parmask = []
#    print(maskedweights)

    startscore = runsim(maskedweights, usemask=False)

    curweight = model.get_npweights()
    tfmask = model.get_npweights()
    cnt = -1
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            cnt += 1
            curweight[i][j] = maskedweights[cnt]
            if usemask:
                tfmask[i][j] = round(float(parmask[cnt]))
    if usemask:
        for i, layer in enumerate(tfmask):
            tfmask[i] = tf.constant(layer)


    model.set_npweights(curweight)
    optimizer = keras.optimizers.Adam(learning_rate=2.5e-3)
    epochs = nepoch
    trainweights = model.get_weights()
    for epoch in range(epochs):
        # Iterate over the batches of the dataset.
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        timestart = time()
        with tf.device('/CPU:0'):
            with tf.GradientTape() as tape:
                tape.watch(trainweights)
                result = model(fusedstruct, atomcount)
                loss_value = mae(result, exactlist, numpyout=False)
            grads = tape.gradient(loss_value, trainweights)
            for i, layer in enumerate(zip(grads, trainweights, tfmask)):
                glayer, wlayer, mlayer = layer
                glayer = tf.math.multiply(glayer, mlayer)
                grads[i] = tf.where(tf.math.is_nan(glayer), 0.0, glayer)
            optimizer.apply_gradients(zip(grads, trainweights))
        timeend = time()
        print("Training loss at step %d: %.4f, time:%s"
                    % (epoch, float(loss_value),timeend-timestart)
                )


    curweight = model.get_npweights()
    cnt = -1
    endparameters = []
    endmasked = []
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            endparameters.append(curweight[i][j])
            endmasked.append(inv_deadzone(curweight[i][j], xlow, xhigh))

    endmasked = np.array(endmasked)
    endmasked = np.where(endmasked == 0.0, parameters, endmasked)
    score = runsim(endmasked, usemask=True)
    endtime = time()
    return score, endmasked
#========================================================
if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    par = []
    with open(filename, 'r') as parfile:
        for line in parfile:
            newpar = float(line.split()[0])
            par.append(newpar)
    score = runsim(par, verbose=True, usemask=False)


