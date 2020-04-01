import pywt
import numpy as np
import time

def gen_wavelet():
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
