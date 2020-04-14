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
def measure_map(mtx):
  pre_proj = np.fft.fft2(run_iDWT(levels, mtx), norm="ortho")
  pre_proj[idxs[:, 0], idxs[:, 1]] = 0
  return pre_proj
## inverse fft mapping. takes the wavelet in the end (need to be modified)
def measure_map_adj(mtx):
  mtx[idxs[:, 0], idxs[:, 1]] = 0
  return run_DWT(levels, np.fft.ifft2(mtx, norm="ortho"))
#gradient of f(x) where L=f(x)+g(x) where g(x) is the sparse term (norm one)
def Sgrad(S,y):
  return measure_map_adj(meas_map(S)-y)

# soft maxing
def prox(S,a):
  return np.exp(1j * np.angle(S)) * np.maximum(np.abs(S) - a, 0)

def lasse_f(S, Y, lambd):
  return 0.5 * np.linalg.norm(meas_map(S) - Y, ord='fro')**2 + lambd * np.sum(np.abs(S))
[p, q] = Y.shape

# loss of lasse it can be modified to momentum gradient later.
def loss_prox(S, Y, lambd,eps):
  # Creating a list to store the loss
  loss = []
  loss.append(lasse_f(S, Y_frob, lambd))
  i = 1
  cond=True
  while(cond):
    S = prox(S - Sgrad(S, Y), lambd) # there is no need for alpha since we have already devided Y by 1/norm
    loss.append(lasse_f(S, Y, lambd))
    if loss[i-1] - loss[i] < eps:
      cond=False
    i = i + 1
    
  return loss, S*Y_frob