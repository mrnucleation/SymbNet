from NetSetUp import getmodel
import numpy as np
import tensorflow as tf
from utils import pretty_print
from time import time

#================================================================================
class ForcefieldModel():
    #---------------------------------------------------------------------------
    def __init__(self, symbasis=None, rcut=8.0, maxatoms=100, offset=0.0 ):
        self.nSymLayers = {'pair':len(symbasis['pair']),
                           'density':len(symbasis['density']),
                           'embedding':len(symbasis['embedding'])}
        self.actfunc = symbasis
        self.pairsym = getmodel(nlayer=self.nSymLayers['pair'], inputdim=1, actfunc=symbasis['pair'])
        self.denssym = getmodel(nlayer=self.nSymLayers['density'], inputdim=1, actfunc=symbasis['density'])
        self.embsym = getmodel(nlayer=self.nSymLayers['embedding'], inputdim=1, actfunc=symbasis['embedding'])
        self.rcut = rcut
        self.rcutsq = rcut*rcut
        self.maxatoms = maxatoms
        self.dummyvec = tf.ones((self.maxatoms,1),dtype=tf.dtypes.float32)
        self.half = tf.constant(0.5)
    #----------------------------------------------------------------------------
    def __call__(self, distlist, atomcount):
        '''
          distlist => A tensor with (M x N_max) dimensions where M = N_max*N_struct
                      Entries are typically fused together for computational
                      efficiency. 
                      N_max is the number of atoms contained in the largest 
                      cluster in the data set.
                      Smaller clusters are padded with zeros to ensure that 
                      all sturctures have a consistent distance matrix size.  
          atomcount => An array with dimension (N_struct) that where each entry
                       contains the number of atoms in the dataset.
        '''


#        print(distlist)
        structcnt = tf.size(distlist)
        structcnt_np = round(int(tf.size(atomcount).numpy()))
        structshape = tf.shape(distlist)
        #Work flow 
        #  Compute p(r_ij) for all structures and atoms
        #  Sum up each row by multiplying the density matrix by v=[1,1,1,1....1]
        #  Feed the density vector into the embedding function
        #  Sum up the resulting vector to give the Embbedding energy per structure.

        outdens = self.denssym(tf.reshape(distlist, shape=(structcnt,1)))
        outdens = tf.reshape(outdens, shape=structshape)
        outdens = tf.where(distlist<1e-32, distlist, outdens)
#        if tf.reduce_any(tf.math.is_nan(outdens)) or tf.reduce_any(tf.math.is_inf(outdens)):
#            print("Nan OutDens")
#            print(distlist)
#            print(outdens)
#            return tf.constant([np.nan]*atomcount)

        denssum = tf.linalg.matmul(outdens, self.dummyvec)
        outemb = self.embsym(denssum)
#        if tf.reduce_any(tf.math.is_nan(outemb)) or tf.reduce_any(tf.math.is_inf(outemb)):
#            print("Nan OutEmb")
#            return tf.constant([np.nan]*len(atomcount))
        #Work flow 
        #  Compute Pair(r_ij) for all structures and atoms
        #  Sum up each row by multiplying the density matrix by v=[1,1,1,1....1]
        #  Sum up the resulting vector to give the Pair energy per structure.
        outpair = self.pairsym(tf.reshape(distlist, shape=(structcnt,1)))
        outpair = tf.reshape(outpair, shape=structshape)

        # Nan Protection for entries that are empty. Zero out
        # any entry where the original distance input was 0.0
        # These entries imply atoms are either out of the cutoff
        # distance or abscent from the system.
        outpair = tf.where(distlist<1e-32, distlist, outpair)
#        if tf.reduce_any(tf.math.is_nan(outpair)) or tf.reduce_any(tf.math.is_inf(outpair)):
#            print("Nan OutPair")
#            return tf.constant([np.nan]*len(atomcount))
        pairsum = tf.linalg.matmul(outpair, self.dummyvec)

        paireng = []
        embeng = []
        nlow = 0
        nhigh = 0
        # Sum up the Pair and Embbed Vector in Strides of N_maxatoms
        for i in range(structcnt_np):
            nlow = nhigh
            nhigh += self.maxatoms
            epair = tf.math.reduce_sum(outpair[nlow:nhigh])
            epair = tf.scalar_mul(self.half, epair)
            paireng.append(epair)
            e_emb = tf.math.reduce_sum(outemb[nlow:nhigh])
            embeng.append(e_emb)

        tf.stack(paireng)
        tf.stack(embeng)

        #Sum the two together and normalize.
        totaleng = tf.math.add(paireng, embeng)
        totaleng = tf.math.divide(totaleng,atomcount)

        return outpair, outdens, outemb, totaleng
    #----------------------------------------------------------------------------
    def buildneighlist(self, atompos):
        natoms = atompos.shape[0]
        neighlist = [list() for iatom in range(natoms)]
        for iAtom, atom in enumerate(atompos):
            if iAtom+1 == natoms:
                break
            distarr = atompos[iAtom+1:] - atom
            distarr = np.square(distarr)
            distarr = np.sum(distarr, axis=1)
            neighs = np.where(distarr <= self.rcutsq)
            neighs = neighs[0] + iAtom+1
            for val in neighs:
                if iAtom == val:
                    continue
                if val not in neighlist[iAtom]:
                    neighlist[iAtom].append(val)
                if iAtom not in neighlist[val]:
                    neighlist[val].append(iAtom)
        for iAtom, neighs in enumerate(neighlist):
            neighs.sort()
        return neighlist
    #----------------------------------------------------------------------------
    def builddistlist(self, atompos, neighlist):
        outpairs = np.zeros(dtype=np.float32,shape=(self.maxatoms,self.maxatoms))
        for iAtom, iCoords in enumerate(atompos):
            if len(neighlist[iAtom]) < 1:
                continue
            for jAtom in neighlist[iAtom]:
                if iAtom <= jAtom:
                    continue
                vij = atompos[jAtom] - atompos[iAtom]
                rij = np.linalg.norm(vij)
                outpairs[iAtom, jAtom] = rij
                outpairs[jAtom, iAtom] = rij
        return outpairs
    #----------------------------------------------------------------------------
    def get_weights(self):
        pairweights = self.pairsym.get_weights()
        densweights = self.denssym.get_weights()
        embweights = self.embsym.get_weights()
        outweights = pairweights + densweights + embweights
        return outweights
    #----------------------------------------------------------------------------
    def get_npweights(self):
        pairweights = self.pairsym.get_npweights()
        densweights = self.denssym.get_npweights()
        embweights = self.embsym.get_npweights()
        outweights = pairweights + densweights + embweights
        return outweights
    #----------------------------------------------------------------------------
    def set_npweights(self, inweights):
        pairweights = inweights[:self.nSymLayers['pair']+1]
        densweights = inweights[self.nSymLayers['pair']+1:self.nSymLayers['pair']+self.nSymLayers['density']+2]
        embweights = inweights[self.nSymLayers['pair']+self.nSymLayers['density']+2:]
        self.pairsym.set_npweights(pairweights)
        self.denssym.set_npweights(densweights)
        self.embsym.set_npweights(embweights)
    #----------------------------------------------------------------------------
    def mask_weights(self, threshold):
        W = self.pairsym.get_npweights()
        for w in W:
            w[np.abs(w) < threshold] = 0
        self.pairsym.set_npweights(W)

        W = self.denssym.get_npweights()
        for w in W:
            w[np.abs(w) < threshold] = 0
        self.denssym.set_npweights(W)

        W = self.embsym.get_npweights()
        for w in W:
            w[np.abs(w) < threshold] = 0
        self.embsym.set_npweights(W)
    #----------------------------------------------------------------------------
    def save_model(self, filename='symmodel.ml'):
        savelist = [self.nSymLayers, self.actfunc, self.rcut,
                     self.pairsym.get_npweights(), self.denssym.get_npweights(),
                     self.embsym.get_npweights()]

        with open(filename, 'wb') as outf:
            pickle.dump(savelist, outf)
    #----------------------------------------------------------------------------
    def load_model(self, filename='symmodel.ml'):
        savelist = pickle.load( open( filename, "rb" ) )

        self.nSymLayers = savelist[0]
        self.actfunc = savelist[1]
        self.rcut = savelist[2]
        self.rcutsq = self.rcut*self.rcut
        self.pairsym = getmodel(nlayer=self.nSymLayers['pair'], inputdim=1, actfunc=self.actfunc['pair'])
        self.pairsym.set_npweights(savelist[3])

        self.denssym = getmodel(nlayer=self.nSymLayers['density'], inputdim=1, actfunc=self.actfunc['density'])
        self.denssym.set_npweights(savelist[4])

        self.embsym = getmodel(nlayer=self.nSymLayers['embedding'], inputdim=1, actfunc=self.actfunc['embedding'])
        self.embsym.set_npweights(savelist[5])
    #----------------------------------------------------------------------------
    def pretty_output(self, threshold=0):
        pair_term = pretty_print.hetrogeneous_network(self.pairsym.get_npweights(), self.actfunc['pair'],
                                                     ['r'], threshold=threshold)
        density_term = pretty_print.hetrogeneous_network(self.denssym.get_npweights(), self.actfunc['density'],
                                                     ['r'], threshold=threshold)
        emb_term = pretty_print.hetrogeneous_network(self.embsym.get_npweights(), self.actfunc['embedding'],
                                                     ['r'], threshold=threshold)
        print('---------------------------------')
        print('Pair term:\n', pair_term)
        print('---------------------------------')
        print('Density term:\n', density_term)
        print('---------------------------------')
        print('Emb. term:\n', emb_term)
        print('---------------------------------')
        return pair_term, density_term, emb_term
 
    #----------------------------------------------------------------------------
