import pywt
import numpy as np
import time
import matplotlib.image as mpimg
from PIL import Image
import scipy
from wav import *

def uniformly_undersample(x):
    """
    returns uniformly undersampled array x
    """
    
    return res

def nonuniformly_undersample(x,num):
    """
    returns nonuniformly undersampled array x
    set num to 1, 2, or 3 to experiment with different non-uniform undersampling schemas
    """
    
    return res

def POCS_input_uniform(x):
    """
    Function to generate POCS core algorithm input with uniform sampling
    1. take fft of image
    2. uniformly undersample
    3. return ifft of undersampled image
    """
    
    F = run_fftc(x)
    F_hat = uniformly_undersample(F)
    res = run_ifftc(F_hat)
    return res

def POCS_input_nonuniform(x, num):
    """
    Function to generate POCS core algorithm input with nonuniform sampling
    1. take fft of image
    2. non uniformly undersample. set num to 1, 2, or 3 to experiment with different non-uniform undersampling schemas
    3. return ifft of undersampled image
    """
    
    F = run_fftc(x)
    F_hat = nonuniformly_undersample(F,num)
    res = run_ifftc(F_hat)
    return res
