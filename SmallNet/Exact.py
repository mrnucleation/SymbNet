import os
from random import random
import sys
import numpy as np

from NNetCalc import runsim
from NNetCalc import activation_funcs, RCUT, maxsize
from FF_Model import ForcefieldModel



dummymodel = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs)
curweight = dummymodel.get_npweights()
parameters = []



curweight[0][0,0] = 0.0
curweight[0][0,1] = 0.0
curweight[0][0,2] = 0.0 
curweight[0][0,3] = 1.0/2.285883762
curweight[1][3,0] = 1.0
curweight[2][0,0] = 1.0

curweight[3][0,4] = 1.0/2.928323832 
curweight[4][4,0] = 1.0

curweight[5][0,2] = 1.0
curweight[6][2,0] = 1.0
curweight[7][0,0] = -1.0

for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        print(i, j, x)
        parameters.append(x)

print(parameters)

score = runsim(parameters, verbose=True, usemask=False)
