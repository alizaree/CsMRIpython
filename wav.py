import pywt
import numpy as np
import time
import matplotlib.image as mpimg
from PIL import Image

def gen_wavelet():
    """
    Generates the wavelet to be used for the 2D DWT and 2D iDWT
    In out code, we use the CDF9/7 wavelet
    """
    
    
    # Define the coefficients for the CDF9/7 filters
    factor=1

    # FORWARD FILTER COEFFICIENTS
    # Forward Decomposition filter: lowpass
    cdf97_an_lo = factor * np.array([0, 0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,
                                     0.602949018236, 0.266864118443, -0.078223266529, -0.016864118443,
                                     0.026748757411])

    # Forward Decomposition filter: highpass
    cdf97_an_hi = factor * np.array([0, 0.091271763114, -0.057543526229, -0.591271763114, 1.11508705,
                                     -0.591271763114, -0.057543526229, 0.091271763114, 0, 0])

    # INVERSE FILTER COEFFICIENTS
    # Inverse Reconstruction filter: lowpass
    cdf97_syn_lo = factor * np.array([0, -0.091271763114, -0.057543526229, 0.591271763114, 1.11508705,
                                      0.591271763114, -0.057543526229, -0.091271763114, 0, 0])

    # Inverse Reconstruction filter: highpass
    cdf97_syn_hi = factor * np.array([0, 0.026748757411, 0.016864118443, -0.078223266529, -0.266864118443,
                                      0.602949018236, -0.266864118443, -0.078223266529, 0.016864118443,
                                      0.026748757411])

    # Create the pywavelets object using the filter coefficients for CDF9/7 filters defined above
    cdf97 = pywt.Wavelet('cdf97', [cdf97_an_lo, cdf97_an_hi, cdf97_syn_lo, cdf97_syn_hi])

    return cdf97

def run_DWT(signal, wav, flag_print=0, mode='zero'):
    """
    Serial implementation of the 2D DWT that also returns the runtime of the program
    :param: signal: input signal to perform 2D DWT
    :param: wav: pywavelet object defined before
    :param: flag_print: whether to print the coefficients or not
    :param: mode: the padding scheme applied to the input (only supports zero-padding)
    :return: cA, cH, cV, cD: 2D DWT coefficients
    :return: time_diff: runtime for the serial program
    """

    # Call the pywavelets 2D DWT function using the pywavelets function
    tic = time.time()
    coeffs = pywt.dwt2(signal, wav, mode)
    toc = time.time()

    cA, (cH, cV, cD) = coeffs
    cA = cA.astype(np.float32)
    cH = cH.astype(np.float32)
    cV = cV.astype(np.float32)
    cD = cD.astype(np.float32)

    time_diff = toc - tic
    if flag_print:
        print("approx: {} \n detail: {} \n{}\n{}\n".format(cA, cH, cV, cD))

    return cA, cH, cV, cD, time_diff

def run_iDWT(wav, cA, cH, cV, cD, mode='zero'):
    """
    Inverse 2D DWT used for reconstructing the original image
    :param: wav: pywavelet object defined before
    :params: cA, cH, cV, cD: 2D DWT coefficients found before
    :param: mode: the padding scheme applied to the input (only supports zero-padding)
    :return: rec_sig: reconstructed image
    """
    coeffs = cA, (cH, cV, cD)
    rec_sig = pywt.idwt2(coeffs, wav, mode)

    return rec_sig

def DWT(signal,wav,levels,mode='zero'):
    """
    returns full scale DWT of signal with multiple levels
    """
    
    coeffs = pywt.wavedec2(signal, wav, mode, level = levels)
    res, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # res is the full dwt image
    # coeff_slices is the list of slices corresponding to each coeffecient
    return res, coeff_slices

def iDWT(s0, wav, coeff_slices, mode='zero'):
    """
    returns full scale iDWT of s0 with multiple levels
    """
    
    coeffs = pywt.array_to_coeffs(s0, coeff_slices, output_format='wavedec2')
    res = pywt.waverec2(coeffs, wav, mode)

    # res is the full recovered image
    return res

def run_fftc(x):
    """
    Performs 2D centered fft on x, returning res
    """
    
    f = np.fft.fft2(x)
    res = np.fft.fftshift(f)
    return res

def run_ifftc(X):
    """
    Performs 2D centered ifft on X, returning res
    """
    
    x_temp = np.fft.ifftshift(X)
    res = np.fft.ifft2(x_temp)
    return res

def rgb2gray(rgb):
    """
    calculates the grayscale of input rgb image
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    res = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return res

def soft_threshold(y,gamma):
    """
    returns thresholded 2D input y according to threshold gamma
    function ignores the phase of input
    """
    
    # set output to zero if y>abs(gamma), otherwise retain value
    res = np.zeros(y.shape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            val = np.absolute(y[i,j])
            if(val>gamma):
                res[i,j] = ((val-gamma)/val)*y[i,j]
    return res
