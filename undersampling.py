import pywt
import numpy as np
import time
import matplotlib.image as mpimg
from PIL import Image
import scipy
from wav import *

def uniformly_undersample(x, fact):
    """
    returns uniformly undersampled array x
    x is undersampled by a factor of fact
    """
    
    res = np.zeros(x.shape)
    for i in range(np.floor(x.shape[0]/fact).astype(np.int32)):
        for j in range(np.floor(x.shape[0]/fact).astype(np.int32)):
            res[i*fact,j*fact] = x[i*fact,j*fact]
    return res

def nonuniformly_undersample(x,num):
    """
    returns nonuniformly undersampled array x
    set num to 1, 2, or 3 to experiment with different non-uniform undersampling schemas
    Schema 1: Pure Random Permutation
    Schema 2: TBD
    Schema 3: TBD
    """
    
    # Random Permutation Schema
    if num == 1:
        res = np.zeros(x.shape)
        for i in range(np.floor(x.shape[0]/fact).astype(np.int32)):
            for j in range(np.floor(x.shape[0]/fact).astype(np.int32)):
                res[i*fact,j*fact] = x[i*fact,j*fact]
    
    return res

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

def POCS_input_nonuniform(f, num):
    """
    Function to generate POCS core algorithm input with nonuniform undersampling
    1. take fft of image
    2. non uniformly undersample. set num to 1, 2, or 3 to experiment with different non-uniform undersampling schemas
    3. return ifft of undersampled image, fft of undersampled image
    """
    
    F = run_fftc(f)
    F_hat = nonuniformly_undersample(F,num)
    f_hat = run_ifftc(F_hat)
    return f_hat, F_hat
