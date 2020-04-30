import pywt
import numpy as np
import time
import matplotlib.image as mpimg
from PIL import Image
import scipy
from wav import *
# changed it to create a map which will later be used in the loss function to recover, changed it so in the map, the elements that are set to 1 will be removed.


## undersampling: return idxs where each idxs(:,0) is the rowss and idxs(:,1) is the columns of pixels which we want to delete and undersample.

def rmvMap_brn( p, sz1,sz2):
    rmvMap = {}# rmvMap will contain all the indices we want to remove.
    for idx1 in np.arange(sz1):
        for idx2 in np.arange(sz2):
            coin = np.random.rand(1)
            if coin > p:
                rmvMap[(idx1, idx2)] = 1
    idxs = np.asarray(list(rmvMap.keys()))
    return idxs



def uniformly_undersample(x, fact):
    """
    returns uniformly undersampled array x
    x is undersampled by factor 'fact'
    """
    
    res = np.ones(x.shape)
    for i in range(np.floor(x.shape[0]/fact).astype(np.int32)):
        for j in range(np.floor(x.shape[0]/fact).astype(np.int32)):
            res[i*fact,j*fact] = 0 #x[i*fact,j*fact]
    idx1, idx2=np.where(res==1)
    idxs=np.vstack((idx1, idx2)).transpose()
    return idxs


def nonuniformly_undersample(x,num,fact):
    """
    returns nonuniformly undersampled array x
    set num to 1, 2, or 3 to experiment with different non-uniform undersampling schemas
    x is undersampled by factor 'fact'
    Schema 1: Pure Random Permutation
    Schema 2: TBD
    Schema 3: TBD
    """
    
    # Random Permutation Schema
    if num == 1:
        res = np.ones(x.shape)
        x = np.random.permutation(x)
        for i in range(np.floor(x.shape[0]/fact).astype(np.int32)):
            for j in range(np.floor(x.shape[0]/fact).astype(np.int32)):
                res[i*fact,j*fact] = 0
    idx1, idx2=np.where(res==1)
    idxs=np.vstack((idx1, idx2)).transpose()
    # TBD Schema
    # TBD Schema
    return idxs

def POCS_input_uniform(f,fact):
    """
    Function to generate POCS core algorithm input with uniform undersampling.
    Undersampling occurs by factor 'fact', which should be a power of 2.
    1. take fft of image
    2. uniformly undersample
    3. return ifft of undersampled image, fft of undersampled image
    """
    
    F = run_fftc(f)
    F_hat = uniformly_undersample(F,fact)
    f_hat = run_ifftc(F_hat)*fact
    return f_hat, F_hat

def POCS_input_nonuniform(f, num, fact):
    """
    Function to generate POCS core algorithm input with nonuniform undersampling
    Undersampling occurs by factor 'fact', which should be a power of 2.
    1. take fft of image
    2. non uniformly undersample. set num to 1, 2, or 3 to experiment with different non-uniform undersampling schemas
    3. return ifft of undersampled image, fft of undersampled image
    """
    
    F = run_fftc(f)
    F_hat = nonuniformly_undersample(F,num, fact)
    f_hat = run_ifftc(F_hat)
    return f_hat, F_hat
