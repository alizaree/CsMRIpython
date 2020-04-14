import os
import nibabel as nib
import awscli as aws
import numpy as np
def download_data(cwd):
    os.chdir(cwd)
    os.mkdir('./content')
    os.chdir('/content')
    os.mkdir('./bold5000')
    os.chdir('/content/bold5000')
    #!aws s3 sync --no-sign-request s3://openneuro.org/ds001499/sub-CSI3/ses-16/anat/ /content/bold5000/sub-CSI3_anat/
def load_data():
    img = nib.load('./content/bold5000/sub-CSI3_anat/sub-CSI3_ses-16_T1w.nii.gz')
    data = img.get_fdata()
    hdr = img.header
    return data,hdr

