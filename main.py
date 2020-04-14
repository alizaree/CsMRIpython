import numpy as np
import os
import nibabel as nib
import matplotlib as mlp
import MainFunctions as MF

# loading the data
data,hdr= MF.load_data()
print(data)