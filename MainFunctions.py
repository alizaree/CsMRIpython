import os
import nibabel as nib
import awscli as aws
import numpy as np
import bokeh
import bokeh.plotting as bpl
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper
import wav as wav

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

## implementing imagesc like in matlab
def imagesc(M, title=''):
    m, n = M.shape

    # 600 px is a good size;
    pw = 600
    #rescale the hight
    ph = round(1.0 * pw * m / n)
    h = bpl.figure(plot_width = pw, plot_height = ph, x_range=(0, 1.0*n),
                 y_range=(0, 1.0*m), toolbar_location='below',
                 title=title, match_aspect=True
                )

    min_pic = np.min(M)
    max_pic = np.max(M)

    color_mapper = LinearColorMapper(palette="Greys256", low=min_pic, high=max_pic)
    h.image(image=[M], x=0, y=0, dw=1.0*n, dh=1.0*m, color_mapper=color_mapper)

    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                      label_standoff=12, border_line_color=None, location=(0, 0))

    h.add_layout(color_bar, 'right')


    bpl.show(h)
    return h
## fft maping ( takes the inverse wavelet then fft (need to be modified)
def rmvMap_brn( p, sz1,sz2):
    rmvMap = {}# rmvMap will contain all the indices we want to remove.
    for idx1 in np.arange(sz1):
        for idx2 in np.arange(sz2):
            coin = np.random.rand(1,)
            rmvMap[(idx1, idx2)] = 1*(coin > p)
    idxs = np.asarray(list(rmvMap.keys()))
    return idxs




def measure_map(mtx,idxs,wave, levels, coeff_slices):
    
    pre_proj = np.fft.fft2( wav.iDWT( mtx, wave,levels, coeff_slices ),norm="ortho")
    #pre_proj= wav.run_ifftc(pre_proj) ## make sure it works make center
    pre_proj[idxs[:, 0], idxs[:, 1]] = 0
    return pre_proj


## inverse fft mapping. takes the wavelet in the end (need to be modified)


def measure_map_adj(mtx,idxs,wave, levels):
    mtx[idxs[:, 0], idxs[:, 1]] = 0
    pre_proj= np.fft.ifft2(mtx ,norm="ortho")# uncenter
    res, dummy = wav.DWT(pre_proj, wave, levels)
    return res



#gradient of f(x) where L=f(x)+g(x) where g(x) is the sparse term (norm one)



def Sgrad(S,y, idxs, wave, levels, coeff_slices):
    
    return measure_map_adj(measure_map(S, idxs, wave, levels, coeff_slices)-y, idxs, wave, levels)



# soft thresholding
def prox(S,a):
    return np.exp(1j * np.angle(S)) * np.maximum(np.abs(S) - a, 0)


def lasse_f(S, Y, lambd, idxs, wave, levels, coeff_slices):
    return 0.5 * np.linalg.norm(measure_map(S, idxs, wave, levels, coeff_slices) - Y, ord='fro')**2 + lambd * np.sum(np.abs(S))



# loss of lasse it can be modified to momentum gradient later.
def loss_prox(S, Y, lambd,eps, idxs, wave, levels, coeff_slices, itr=float("inf"), minItr=0):
    # Creating a list to store the loss
    Y_frob = np.linalg.norm(Y, ord='fro')
    Y_scaled = Y / Y_frob
    alpha=1
    loss = []
    loss.append(lasse_f(S, Y_frob, lambd, idxs, wave, levels, coeff_slices))
    i = 1
    cond=True
    while(cond):
        S = prox(S - Sgrad(S, Y_scaled, idxs, wave, levels, coeff_slices), lambd)
        loss.append(lasse_f(S, Y_scaled, lambd, idxs, wave, levels, coeff_slices))
        if i>minItr:
            if (loss[i-1] - loss[i]) < eps or itr<i:
                cond=False
        i = i + 1

    return loss, S

def meas_fun(mtx,idxs,levels, wave=None):
    pre_proj = np.fft.fft2(wav.iDWT( mtx, wave,levels), norm="ortho")
    pre_proj[idxs[:, 0], idxs[:, 1]] = 0
    return pre_proj


def meas_fun_adj(mtx,idxs,levels, wave=None):
    mtx[idxs[:, 0], idxs[:, 1]] = 0
    return wav.DWT( np.fft.ifft2(mtx, norm="ortho"), wave,levels)

def Sgrad2(S,y,idxs,levels):
    return meas_fun_adj(meas_fun(S,idxs,levels)-y,idxs,levels)

def prox2(S,a):
    return np.exp(1j * np.angle(S)) * np.maximum(np.abs(S) - a, 0)

def lasse_f2(S, Y, lambd,idxs,levels):
    return 0.5 * np.linalg.norm(meas_fun(S,idxs,levels) - Y, ord='fro')**2 + lambd*np.sum(np.abs(S)) 


def loss_prox2(S, Y, lambd,eps,Y_frob,idxs,levels,itr ):
  # Creating a list to store the loss
    loss = []
    loss.append(lasse_f2(S, Y_frob, lambd,idxs,levels))
    i = 1
    cond=True
    while(cond):
        S = prox2(S - Sgrad2(S, Y,idxs,levels), lambd) # there is no need for alpha since we have already devided Y by 1/norm
        loss.append(lasse_f2(S, Y, lambd,idxs,levels))
        if loss[i-1] - loss[i] < eps or i>itr:
            cond=False
        i = i + 1
    
    return loss, S*Y_frob